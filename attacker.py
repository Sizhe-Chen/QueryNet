from surrogate import *

def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1
    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
        max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1
    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
    return delta

class QueryNet():
    def __init__(self, sampler, victim_name, surrogate_names, use_square_plus, use_square, use_nas, l2_attack, eps, batch_size):
        self.surrogate_names = surrogate_names
        self.use_surrogate_attacker = self.surrogate_names != []
        self.use_square_plus = use_square_plus
        self.use_square = use_square
        assert (self.use_surrogate_attacker     and self.use_square_plus     and self.use_square) or \
               (self.use_surrogate_attacker     and not self.use_square_plus and self.use_square) or \
               (self.use_surrogate_attacker     and not self.use_square_plus and not self.use_square) or \
               (not self.use_surrogate_attacker and self.use_square_plus     and not self.use_square) or \
               (not self.use_surrogate_attacker and not self.use_square_plus and self.use_square)
        self.eps = eps
        self.use_nas = use_nas

        self.train_loss_thres = 2 if ('easydl' not in victim_name) else 0.025 # easydl outputs prob instead of logits
        self.batch_size = batch_size
        self.victim_name = victim_name
        self.square_plus_max_trial = 50
        self.surrogate_train_iter = [30, 100, 1500] 
        # stop training if the training loss < self.train_loss_thres after self.surrogate_train_iter[0] batches or anycase after self.surrogate_train_iter[1]
        # if the surrogate is not externally trained and save in ./pretrained by query pairs from the first two queries
        self.save_surrogate = True
        self.save_surrogate_path = sampler.result_dir + '/srg'
        os.makedirs(self.save_surrogate_path)
        
        self.sampler = sampler
        self.generator = PGDGeneratorInfty(int(batch_size / 2)) if not l2_attack else PGDGenerator2(int(batch_size / 2))
        self.square_attack = self.square_attack_linfty if not l2_attack else self.square_attack_l2
        self.surrogates = []
        os.makedirs('pretrained', exist_ok=True)
        gpus = torch.cuda.device_count()
        num_class = self.sampler.label.shape[1]
        
        self.use_nas_layers = [10, 6, 8, 4, 12, 14]
        if self.sampler.data.shape[1] == 1: self.use_nas_layers = [int(x/2) for x in self.use_nas_layers] # use smaller search space for MNIST
        self.loaded_trained_surrogates_on_past_queries = []
        self.surrogate_save_paths = []
        for i, surrogate_name in enumerate(surrogate_names):
            if not self.use_nas: self.surrogates.append(Surrogate(surrogate_name, num_class=num_class, softmax='easydl' in victim_name,
                gpu_id=0 if gpus==1 else i % (gpus-1)+1)) 
            else:            self.surrogates.append(NASSurrogate(init_channels=16, layers=self.use_nas_layers[i], num_class=num_class, n_channels=self.sampler.data.shape[1], softmax='easydl' in victim_name,
                gpu_id=0 if gpus==1 else i % (gpus-1)+1)) 
            
            save_info = 'pretrained/netSTrained_{}_{}_{}.pth'.format(surrogate_name, victim_name, 0) if not self.use_nas else \
                        'pretrained/netSTrained_NAS{}_{}_latest.pth'.format(self.use_nas_layers[i], self.victim_name)
            
            self.surrogate_save_paths.append(save_info)
            if os.path.exists(save_info): 
                self.surrogates[i].load(save_info)
                self.loaded_trained_surrogates_on_past_queries.append(True)
            else:
                self.loaded_trained_surrogates_on_past_queries.append(False)
         
        self.num_attacker = len(surrogate_names) + int(use_square_plus) + int(use_square)
        self.attacker_eva_weights = [1] * len(surrogate_names) + [0, 0] 
        if self.sampler.data.shape[1] == 1: self.attacker_eva_weights[-1] = 0

        if num_class == 1000: self.eva_weights_zero_threshold = 20 # set evaluation to zero if the denominator is small
        elif self.sampler.data.shape[1] == 1: self.eva_weights_zero_threshold = 2
        else: self.eva_weights_zero_threshold = 10
    
    def meta_pseudo_gaussian_pert(self, s):
        delta = np.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
            if np.random.rand(1) > 0.5: delta = np.transpose(delta)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        return delta

    def square_attack_l2(self, x_curr, x_best_curr, deltas, is_candidate_maximizer, min_val, max_val, p, **kwargs):
        c, h, w = x_curr.shape[1:]
        n_features = c * h * w
        s = max(int(round(np.sqrt(p * n_features / c))), 3)
        if s % 2 == 0: s += 1
        s2 = s + 0
        
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0

        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((deltas * mask_2) ** 2, axis=(2, 3), keepdims=True))

        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = deltas[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                np.maximum(self.eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        deltas[~is_candidate_maximizer, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        deltas[~is_candidate_maximizer, :, center_h:center_h + s, center_w:center_w + s] = new_deltas[~is_candidate_maximizer, ...] + 0  # update window_1


        x_new = x_curr + deltas / np.sqrt(np.sum(deltas ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps
        x_new = np.clip(x_new, min_val, max_val)
        return x_new, deltas
    
    def square_attack_linfty(self, x_curr, x_best_curr, deltas, is_candidate_maximizer, min_val, max_val, p, **kwargs):
        c, h, w = x_curr.shape[1:]
        n_features = c * h * w
        s = int(round(np.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        deltas[~is_candidate_maximizer, :, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-self.eps, self.eps], size=[c, 1, 1])

        # judge overlap
        for i_img in range(x_best_curr.shape[0]):
            if is_candidate_maximizer[i_img]: continue
            center_h_tmp, center_w_tmp, s_tmp = center_h, center_w, s
            while np.sum(np.abs(np.clip(
                x_curr[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp] + 
                deltas[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp], 
                min_val, max_val) - 
                x_best_curr[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp]) 
                < 10 ** -7) == c * s * s:
                s_tmp = int(round(np.sqrt(p * n_features / c)))
                s_tmp = min(max(s_tmp, 1), h - 1) 
                center_h_tmp, center_w_tmp = np.random.randint(0, h - s_tmp), np.random.randint(0, w - s_tmp)
                deltas[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp] = np.random.choice([-self.eps, self.eps], size=[c, 1, 1])
        return np.clip(x_curr + deltas, min_val, max_val), deltas
    
    def square_attacker(self, x_curr, x_best_curr, **kwargs): 
        x_next, _ = self.square_attack(x_curr, x_best_curr, x_best_curr-x_curr, np.zeros(x_best_curr.shape[0], dtype=np.bool), **kwargs)
        return x_next

    def square_plus_attacker(self, x_curr, x_best_curr, **kwargs):
        is_candidate_maximizer = np.zeros(x_best_curr.shape[0], dtype=np.bool)
        deltas = x_best_curr-x_curr
        for i in range(self.square_plus_max_trial): # retry random search for a maximum of self.square_plus_max_trial times
            x_next, deltas = self.square_attack(x_curr, x_best_curr, deltas, is_candidate_maximizer, **kwargs)
            is_candidate_maximizer = self.sampler.judge_potential_maximizer(x_next)
            if np.sum(is_candidate_maximizer) == x_best_curr.shape[0]: break
        return x_next
    
    def surrogate_attacker(self, x_curr, x_best_curr, y_curr, attacker_id, targeted):#, **kwargs):
        assert attacker_id < len(self.surrogate_names)
        os.makedirs(self.sampler.result_dir + '/log', exist_ok=True)
        log_file_path = '%s/log/surrogate%d_train.log' % (self.sampler.result_dir, attacker_id)
        for i in range(self.surrogate_train_iter[1] if self.loaded_trained_surrogates_on_past_queries[attacker_id] else self.surrogate_train_iter[2]):
            train_loss = self.surrogates[attacker_id].train(attacker_id, self.sampler, self.batch_size, i, log_file_path=log_file_path)
            if train_loss < self.train_loss_thres and i > self.surrogate_train_iter[0]: break # train surrogate until convergence
            if not self.loaded_trained_surrogates_on_past_queries[attacker_id]:
                self.surrogates[attacker_id].save(self.surrogate_save_paths[attacker_id]) 
                # if the surrogates are not pretrained on first 2 iteration queries, conduct thorough training and save it for later usage
                self.loaded_trained_surrogates_on_past_queries[attacker_id] = True
        
        iter_trained = 0
        while 1:
            if not self.use_nas: save_info = '{}/netSTrained_{}_{}_{}.pth'.format(self.save_surrogate_path, self.surrogate_names[attacker_id], self.victim_name, iter_trained)
            else: save_info = '{}/netSTrained_NAS{}_{}_{}.pth'.format(self.save_surrogate_path, self.use_nas_layers[attacker_id], self.victim_name, iter_trained)
            if not os.path.exists(save_info): break
            iter_trained += 1
        if self.save_surrogate: self.surrogates[attacker_id].save(save_info)
        # FGSM attack
        self.x_new_tmp[attacker_id] = self.generator(x_best_curr, x_curr, self.eps, self.surrogates[attacker_id], y_curr, targeted=targeted)

    def surrogate_attacker_multi_threading(self, x_curr, x_best_curr, y_curr, targeted, **kwargs):
        threads = [] # train and attack via different surrogates simultaneously
        self.x_new_tmp = [0 for _ in range(len(self.surrogate_names))]
        for attacker_id in range(len(self.surrogate_names)):
            threads.append(threading.Thread(target=self.surrogate_attacker, args=(x_curr, x_best_curr, y_curr, attacker_id, targeted)))
        for attacker_id in range(len(self.surrogate_names)): threads[attacker_id].start()
        for attacker_id in range(len(self.surrogate_names)): 
            if threads[attacker_id].isAlive(): threads[attacker_id].join()
        return self.x_new_tmp

    def yield_candidate_queries(self, x_curr, x_best_curr, y_curr, **kwargs):
        if max(self.attacker_eva_weights) == self.attacker_eva_weights[-2]: # max(w_{1~n}) < w_{n+1}, w_{n+2} < w_{n+1}
            x_new_candidate = []
            if self.use_square_plus:  x_new_candidate.append(self.square_plus_attacker(x_curr, x_best_curr, **kwargs))
            if self.use_square:      x_new_candidate.append(self.square_attacker(x_curr, x_best_curr, **kwargs))
            return x_new_candidate
        elif max(self.attacker_eva_weights) == self.attacker_eva_weights[-1]: # max(w_{1~n}) < w_{n+1} < w_{n+2}
            return [self.square_attacker(x_curr, x_best_curr, **kwargs)]
        else: # max(w_{1~n}) > w_{n+1}
            x_new_candidate = self.surrogate_attacker_multi_threading(x_curr, x_best_curr, y_curr, **kwargs)
            if self.use_square_plus: x_new_candidate.append(self.square_plus_attacker(x_curr, x_best_curr, **kwargs))
            elif self.use_square:   x_new_candidate.append(self.square_attacker(x_curr, x_best_curr, **kwargs))
            return x_new_candidate

    def forward(self, x_curr, x_best_curr, y_curr, get_surrogate_loss, **kwargs):
        x_new_candidate = self.yield_candidate_queries(x_curr, x_best_curr, y_curr, **kwargs)
        if len(x_new_candidate) == 1: return x_new_candidate[0], None # max(w_{1~n}) < w_{n+1} < w_{n+2}, use square only
        else:
            loss_candidate = [] # num_attacker * num_sample
            for attacker_id in range(len(x_new_candidate)):
                loss_candidate_for_one_attacker = []
                for evaluator_id in range(len(self.surrogate_names)):
                    loss_candidate_for_one_attacker.append(
                        get_surrogate_loss(self.surrogates[evaluator_id], x_new_candidate[attacker_id], y_curr) * self.attacker_eva_weights[evaluator_id]
                    ) 
                loss_candidate.append(sum(loss_candidate_for_one_attacker)/len(loss_candidate_for_one_attacker))
            loss_candidate = np.array(loss_candidate)
            
            x_new_index = np.argmin(loss_candidate, axis=0) # a, selected attacker IDs
            x_new = np.zeros(x_curr.shape) # x^q
            for attacker_id in range(len(x_new_candidate)):
                attacker_index = x_new_index == attacker_id
                x_new[attacker_index] = x_new_candidate[attacker_id][attacker_index]
        return x_new, x_new_index
    
    def backward(self, idx_improved, x_new_index, **kwargs):
        if self.use_surrogate_attacker: 
            if self.use_square_plus or self.use_square:
                save_only = max(self.attacker_eva_weights) == self.attacker_eva_weights[-1]
                self.sampler.update_buffer(save_only=save_only, **kwargs)
                if x_new_index is not None and self.use_square_plus and not save_only: self.sampler.update_lipschitz()
            else:
                self.sampler.update_buffer(save_only=False, **kwargs) # FGSM and Square+ require to update buffer
        elif self.use_square_plus: # Square+ only, no surrogates
            self.sampler.update_buffer(save_only=False, **kwargs)
            self.sampler.update_lipschitz() # only do this for square+
        elif self.use_square: # Square only, no surrogates
            self.sampler.update_buffer(save_only=True, **kwargs)

        if x_new_index is None: return None # Square, do nothing
        
        elif max(self.attacker_eva_weights) == self.attacker_eva_weights[-2]: # Square+, Square
            assert x_new_index.max() == 1 and self.use_square_plus and self.use_square # only valid when they are both adopted
            attacker_selected = [0 for _ in range(len(self.surrogate_names))]
            for attacker_id in range(x_new_index.max()+1):

                attacker_index = x_new_index == attacker_id
                attacker_selected.append(np.mean(attacker_index))
                attacker_id_real = attacker_id + len(self.surrogate_names)
                
                if np.sum(attacker_index) < self.eva_weights_zero_threshold: self.attacker_eva_weights[attacker_id_real] = 0  #21.5.4  few samples fail to judge the eva_weights
                else: self.attacker_eva_weights[attacker_id_real] = np.sum(idx_improved[attacker_index]) / np.sum(attacker_index)

        else: 
            # (1) FGSM, Square+ in QueryNet (A = {FGSM, Square+, Square})
            # (2) FGSM, Square if we do not include Square+ (A = {FGSM, Square})
            # (3) FGSM if A = {FGSM}
            # (4) Square+ if A = {Square+}
            assert x_new_index.max() in [len(self.surrogate_names)-1, len(self.surrogate_names)]
            attacker_selected = []
            for attacker_id in range(x_new_index.max()+1):

                attacker_index = x_new_index == attacker_id
                if x_new_index.max() == len(self.surrogate_names)-1 or attacker_id != x_new_index.max():
                    #                                           (3) or attacker_id is not the last, no need to handle attacker_selected exceptionally
                    attacker_id_real = attacker_id
                    attacker_selected += [np.mean(attacker_index)]
                    if attacker_id == x_new_index.max(): attacker_selected += [0, 0] # (3)
                elif self.use_square_plus: # attacker_id is the last, (1), (4)
                    attacker_id_real = attacker_id
                    attacker_selected += [np.mean(attacker_index), 0] # no Square in these cases
                else: # attacker_id is the last, (2)
                    attacker_id_real = attacker_id + 1 # no Square+ in this case, so the last index of x_new is actually for Square
                    attacker_selected += [0, np.mean(attacker_index)]

                if np.sum(attacker_index) < self.eva_weights_zero_threshold: self.attacker_eva_weights[attacker_id_real] = 0
                else: self.attacker_eva_weights[attacker_id_real] = np.sum(idx_improved[attacker_index]) / np.sum(attacker_index)
        

        field_names = ['ATTACK']
        if not self.use_nas: field_names += self.surrogate_names
        else: field_names += ['NASlayer' + str(l) for l in self.use_nas_layers[:len(self.surrogate_names)]]
        field_names += ['Square+', 'Square']
        tb = pt.PrettyTable()
        tb.field_names = field_names
        width = {}
        for i, field_name in enumerate(field_names):
            if i: width[field_name] = 11
        tb._min_width = width
        tb._max_width = width
        tb.add_row(['WEIGHT'] + ['%.3f' % x for x in self.attacker_eva_weights])
        tb.add_row(['CHOSEN'] + ['%.3f' % x for x in attacker_selected])
        return str(tb)
        

class PGDGeneratorInfty():
    def __init__(self, max_batch_size):
        self.device = torch.device('cpu')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_batch_size = max_batch_size

    def _call(self, img, lbl, surrogate, epsilon, targeted):
        # img : B * H * W * C  0~1 np.float32 array
        img = img.to(surrogate.device)
        img.requires_grad = True
        lbl = torch.Tensor(lbl).to(surrogate.device)

        alpha = epsilon * 2
        num_iter = 1

        momentum_grad = 0
        for i in range(num_iter):
            img.requires_grad = True
            loss = self.criterion(surrogate(img, no_grad=False), lbl.argmax(dim=-1))
            surrogate.surrogate.zero_grad()
            loss.backward()
            grad = img.grad.data
            momentum_grad += grad
            img = img + alpha * momentum_grad.sign() # maximum attack step: FGSM
        return img.to(self.device)

    def __call__(self, img, ori, epsilon, surrogate, lbl, return_numpy=True, targeted=False):
        # img : B * H * W * C  0~1 np.float32 array
        # return: B * H * W * C  np.float32 array   /   B * C * H * W  0~1  torch.Tensor
        # CPU
        torch.cuda.empty_cache()
        img, ori = torch.Tensor(img), torch.Tensor(ori)
        batch_size = min([self.max_batch_size, img.shape[0]])
        if batch_size < self.max_batch_size: adv = self._call(img, lbl, surrogate, epsilon, targeted=targeted)
        else:
            batch_num = int(img.shape[0] / batch_size)
            if batch_size * batch_num != int(img.shape[0]): batch_num += 1
            adv = self._call(img[:batch_size], lbl[:batch_size], surrogate, epsilon, targeted=targeted)
            for i in range(batch_num-1):
                new_adv = torch.cat((adv, 
                                self._call(img[batch_size*(i+1):batch_size*(i+2)], 
                                           lbl[batch_size*(i+1):batch_size*(i+2)], 
                                           surrogate, epsilon, targeted=targeted)), 0) 
                adv = new_adv
        
        adv = torch.min(torch.max(adv, ori - epsilon), ori + epsilon)
        adv = torch.clamp(adv, 0.0, 1.0)
        if return_numpy: return adv.detach().numpy()
        else: return adv


class PGDGenerator2():
    def __init__(self, max_batch_size):
        self.device = torch.device('cpu')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_batch_size = max_batch_size

    def _call(self, img, ori, lbl, surrogate, epsilon, targeted):
        # img : B * H * W * C  0~1 np.float32 array
        img = img.to(surrogate.device)
        img.requires_grad = True
        lbl = torch.Tensor(lbl).to(surrogate.device)
        
        alpha = epsilon * 2
        
        loss = self.criterion(surrogate(img, no_grad=False), lbl.argmax(dim=-1))
        surrogate.surrogate.zero_grad()
        loss.backward()
        grad = img.grad.data
        img = img + alpha * grad / \
            torch.norm(grad.reshape(grad.shape[0], -1), dim=1, p=2, keepdim=True).reshape(-1).repeat(grad.shape[1], grad.shape[2], grad.shape[3], 1).permute(3, 0, 1, 2)
        return img.to(self.device)

    def __call__(self, img, ori, epsilon, surrogate, lbl, return_numpy=True, targeted=False):
        # img : B * H * W * C  0~1 np.float32 array
        # return: B * H * W * C  np.float32 array   /   B * C * H * W  0~1  torch.Tensor
        # CPU
        torch.cuda.empty_cache()
        img, ori = torch.Tensor(img), torch.Tensor(ori)
        batch_size = min([self.max_batch_size, img.shape[0]])
        if batch_size < self.max_batch_size: adv = self._call(img, ori, lbl, surrogate, epsilon, targeted=targeted)
        else:
            batch_num = int(img.shape[0] / batch_size)
            if batch_size * batch_num != int(img.shape[0]): batch_num += 1
            adv = self._call(img[:batch_size], ori[:batch_size], lbl[:batch_size], surrogate, epsilon, targeted=targeted)
            for i in range(batch_num-1):
                #print(i, batch_num, end='\r')
                new_adv = torch.cat((adv, 
                                self._call(img[batch_size*(i+1):batch_size*(i+2)], 
                                           ori[batch_size*(i+1):batch_size*(i+2)], 
                                           lbl[batch_size*(i+1):batch_size*(i+2)], 
                                           surrogate, epsilon, targeted=targeted)), 0) 
                adv = new_adv
        
        per = adv - ori
        adv = ori + per / \
            torch.norm(per.reshape(per.shape[0], -1), dim=1, p=2, keepdim=True).reshape(-1).repeat(per.shape[1], per.shape[2], per.shape[3], 1).permute(3, 0, 1, 2) * epsilon
        adv = torch.clamp(adv, 0.0, 1.0)
        if return_numpy: return adv.detach().numpy()
        else: return adv
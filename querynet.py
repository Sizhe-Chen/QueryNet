from attacker import *
from victim import *


def p_selection(p_init, it, num_iter):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / num_iter * 10000)
    if   10 < it <= 50:       return p_init / 2
    elif 50 < it <= 200:      return p_init / 4
    elif 200 < it <= 500:     return p_init / 8
    elif 500 < it <= 1000:    return p_init / 16
    elif 1000 < it <= 2000:   return p_init / 32
    elif 2000 < it <= 4000:   return p_init / 64
    elif 4000 < it <= 6000:   return p_init / 128
    elif 6000 < it <= 8000:   return p_init / 256
    elif 8000 < it <= 10000:  return p_init / 512
    else:                     return p_init


def attack(model, x, y, logits_clean, dataset, batch_size, run_time, args):
    eps, seed, l2_attack, num_iter, p_init, num_srg, use_square_plus, use_nas = \
        (args.eps / 255 if not args.l2_attack else args.eps), (args.seed if args.seed != -1 else run_time), \
        args.l2_attack, args.num_iter, args.p_init, args.num_srg, args.use_square_plus, args.use_nas
    np.random.seed(seed)
    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    n_features = c * h * w

    if l2_attack: # the initial stripes in square attack
        delta_init = np.zeros(x.shape)
        s = h // 5
        sp_init = (h - s * 5) // 2
        center_h = sp_init + 0
        for counter in range(h // s):
            center_w = sp_init + 0
            for counter2 in range(w // s):
                delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += QueryNet.meta_pseudo_gaussian_pert(None, s).reshape(
                    [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
                center_w += s
            center_h += s
        x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * eps, 0, 1)
    else:
        x_best = np.clip(x + np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w]), min_val, max_val)

    logits = model(x_best)
    loss_min = get_margin_loss(y, logits)
    n_queries = np.ones(x.shape[0]) * 2  # have queried with original samples and stripe samples

    surrogate_names = ['DenseNet121', 'ResNet50', 'DenseNet169', 'ResNet101', 'DenseNet201', 'VGG19'][:num_srg] # surrogates if not using nas
    result_path = get_time() + f'_{dataset}_{model.arch}' + \
        ('_l2' if l2_attack else '_linfty') + \
        f'_eps{round(eps* (255 if not l2_attack else 1), 2)}' + \
        ('_Eval' if num_srg != 0 else '') + \
        ('_Sqr+' if use_square_plus else '') + \
        (f'_NAS{num_srg}' if use_nas else ('_'+'-'.join(surrogate_names) if len(surrogate_names) != 0 else ''))
    print(result_path)
    logger = LoggerUs(result_path)
    os.makedirs(result_path + '/log', exist_ok=True)
    log.reset_path(result_path + '/log/main.log')
    metrics_path = logger.result_paths['base'] + '/log/metrics'
    log.print(str(args))

    sampler = DataManager(x, logits_clean, eps, result_dir=result_path, loss_init=get_margin_loss(y, logits_clean))
    sampler.update_buffer(x_best, logits, loss_min, logger, targeted=False, data_indexes=None, margin_min=loss_min)
    sampler.update_lipschitz()
    querynet = QueryNet(sampler, model.arch, surrogate_names, use_square_plus, True, use_nas, l2_attack, eps, batch_size)

    def get_surrogate_loss(srgt, x_adv, y_ori): # for transferability evaluation in QueryNet's 2nd forward operation
        if x_adv.shape[0] <= batch_size:  return get_margin_loss(y_ori, srgt(torch.Tensor(x_adv)).cpu().detach().numpy())
        batch_num = int(x_adv.shape[0]/batch_size)
        if batch_size * batch_num != int(x_adv.shape[0]): batch_num += 1
        loss_value = get_margin_loss(y_ori[:batch_size], srgt(torch.Tensor(x_adv[:batch_size])).cpu().detach().numpy())
        for i in range(batch_num-1):
            new_loss_value = get_margin_loss(y_ori[batch_size*(i+1):batch_size*(i+2)], srgt(torch.Tensor(x_adv[batch_size*(i+1):batch_size*(i+2)])).cpu().detach().numpy())
            loss_value = np.concatenate((loss_value, new_loss_value), axis=0)
            del new_loss_value
        return loss_value

    time_start = time.time()
    metrics = np.zeros([num_iter, 7])
    for i_iter in range(num_iter):
        # focus on unsuccessful AEs
        idx_to_fool = loss_min > 0
        x_curr, x_best_curr, y_curr, loss_min_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool], loss_min[idx_to_fool]

        # QueryNet's forward propagation
        x_q, a = querynet.forward(x_curr, x_best_curr, y_curr, get_surrogate_loss, min_val=min_val, max_val=max_val, p=p_selection(p_init, i_iter, num_iter), targeted=False)

        # query
        logits = model(x_q)
        loss = get_margin_loss(y_curr, logits)
        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_q + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        # QueryNet's backward propagation
        message = querynet.backward(idx_improved, a, data_indexes=np.where(idx_to_fool)[0], 
            margin_min=loss_min, img_adv=x_q, lbl_adv=logits, loss=loss, logger=logger, targeted=False)
        if a is not None: 
            print(' '*80, end='\r')
            log.print(message)
            querynet.sampler.save(i_iter)

        # logging
        acc_corr = (loss_min > 0.0).mean()
        mean_nq_all, mean_nq = np.mean(n_queries), np.mean(n_queries[loss_min <= 0])
        median_nq_all, median_nq = np.median(n_queries)-1, np.median(n_queries[loss_min <= 0])-1
        avg_loss = np.mean(loss_min)
        elapse = time.time() - time_start
        msg = '{}: Acc={:.2%}, AQ_suc={:.2f}, MQ_suc={:.1f}, AQ_all={:.2f}, MQ_all={:.1f}, ALoss_all={:.2f}, |D|={:d}, Time={:.1f}s'.\
            format(i_iter + 1, acc_corr, mean_nq, median_nq, mean_nq_all, median_nq_all, avg_loss, querynet.sampler.clean_sample_indexes[-1], elapse)
        log.print(msg if 'easydl' not in model.arch else msg + ', query=%d' % model.query)
        metrics[i_iter] = [acc_corr, mean_nq, median_nq, mean_nq_all, median_nq_all, avg_loss, elapse]
        np.save(metrics_path, metrics)
        if acc_corr == 0: break
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', default='gdas', type=str,
                        help='network architecture [wrn-28-10-drop, gdas, pyramidnet272, easydlcifar] for CIFAR10'
                        '[inception_v3, mnasnet1_0, resnext101_32x8d] for ImageNet'
                        '[resnet_preact, wrn, densenet, easydlmnist] for MNIST')
    parser.add_argument('--l2_attack', action='store_true', help='perform l2 attack')
    parser.add_argument('--eps', type=float, default=16, help='the attack bound')
    parser.add_argument('--num_iter', type=int, default=10000, help='maximum query times.')

    parser.add_argument('--num_x', type=int, default=10000, help='number of samples for evaluation.')
    parser.add_argument('--num_srg', type=int, default=0, help='number of surrogates.')
    parser.add_argument('--use_nas', action='store_true', help='use NAS to train the surrogate.')
    parser.add_argument('--use_square_plus', action='store_true', help='use Square+.')
    
    parser.add_argument('--p_init', type=float, default=0.05, help='hyperparameter of Square, the probability of changing a coordinate.')
    parser.add_argument('--gpu', type=str, default='1', help='GPU number(s).')
    parser.add_argument('--run_times', type=int, default=1, help='repeated running time.')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    args = parser.parse_args()
    if args.use_nas: assert args.num_srg > 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    log = Logger('')
    
    for model_name in args.model.split(','):
        if   model_name in ['wrn-28-10-drop', 'gdas', 'pyramidnet272', 'easydlcifar']: dataset = 'cifar10'
        elif model_name in ['inception_v3', 'mnasnet1_0', 'resnext101_32x8d']:         dataset = 'imagenet'
        elif model_name in ['resnet_preact', 'wrn', 'densenet', 'easydlmnist']:        dataset = 'mnist'
        else: raise ValueError('Invalid Victim Name!')

        if dataset == 'mnist':
            if not ((args.l2_attack and args.eps == 3) or (not args.l2_attack and args.eps == 76.5)): 
                print('Warning: not using default eps in the paper, which is l2=3 or linfty=76.5 for MNIST.')
            x_test, y_test = load_mnist(args.num_x)
            batch_size = 3000 if not args.use_nas else 300
            model = VictimMnist(model_name, batch_size=batch_size)
        
        elif dataset == 'imagenet':
            assert (not args.use_nas), 'NAS is not supported for ImageNet for resource concerns'
            if not ((args.l2_attack and args.eps == 5) or (not args.l2_attack and args.eps == 12.75)): 
                print('Warning: not using default eps in the paper, which is l2=5 or linfty=12.75 for ImageNet.')
            batch_size = 100 if model_name != 'resnext101_32x8d' else 32
            model = VictimImagenet(model_name, batch_size=batch_size) if model_name != 'easydlmnist' else VictimEasydl(arch='easydlmnist')
            x_test, y_test = load_imagenet(args.num_x, model)
        
        else:
            if not ((args.l2_attack and args.eps == 3) or (not args.l2_attack and args.eps == 16)): 
                print('Warning: not using default eps in the paper, which is l2=3 or linfty=16 for CIFAR10.')
            x_test, y_test = load_cifar10(args.num_x)
            batch_size = 2048 if not args.use_nas else 128
            model = VictimCifar(model_name, no_grad=True, train_data='full', epoch='final').eval() if model_name != 'easydlcifar' else VictimEasydl(arch='easydlcifar')
            
        logits_clean = model(x_test)
        corr_classified = logits_clean.argmax(1) == y_test.argmax(1)
        print('Clean accuracy: {:.2%}'.format(np.mean(corr_classified)) + ' ' * 40)
        y_test = dense_to_onehot(y_test.argmax(1), n_cls=10 if dataset != 'imagenet' else 1000)
        for run_time in range(args.run_times):
            attack(model, x_test[corr_classified], y_test[corr_classified], logits_clean[corr_classified], dataset, batch_size, run_time, args)

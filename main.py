import numpy as np
import tensorflow as tf
import datetime, time, scipy.io
from model import *
from util import *

# --------------------------------- HYPER-PARAMETERS --------------------------------- #
in_channels = 1
out_channels = 1
n_epochs = 150
batch_size = 1
learning_rate = 0.0002
beta1 = 0.9

display_steps = 200
save_epochs = 10
src_suffix = 'raw'
dst_suffix = 'target'


def gen_list(data_dir):
    file_list = glob.glob(os.path.join(data_dir, src_suffix, '*.*'))
    file_list.sort()
    file_pair_list = []
    for path1 in file_list:
        path2 = path1.replace(src_suffix, dst_suffix)
        path12 = path1 + ' ' + path2
        file_pair_list.append(path12)
    return file_pair_list


def train(train_list, val_list, debug_mode=True):
    print('Running PRLNet -Training!')
    # create folders to save trained model and results
    graph_dir   = './graph'
    checkpt_dir = './checkpoints'
    ouput_dir   = './output'
    exists_or_mkdir(graph_dir, need_remove=True)
    exists_or_mkdir(ouput_dir)
    exists_or_mkdir(checkpt_dir)

    # --------------------------------- load data ---------------------------------
    # data fetched at range: [-1,1]
    input_imgs, target_imgs, num = input_producer(train_list, in_channels, batch_size, need_shuffle=True)
    if debug_mode:
        input_val, target_val, num_val = input_producer(val_list, in_channels, batch_size, need_shuffle=False)

    pred_content, pred_detail, pred_imgs = gen_PRLNet(input_imgs, out_channels, is_train=True, reuse=False)
    if debug_mode:
        _, _, pred_val = gen_PRLNet(input_val, out_channels, is_train=False, reuse=True)

    # --------------------------------- loss terms ---------------------------------
    with tf.name_scope('Loss') as loss_scp:
        target_224 = tf.image.resize_images(target_imgs, size=[224, 224], method=0, align_corners=False)
        predict_224 = tf.image.resize_images(pred_imgs, size=[224, 224], method=0, align_corners=False)
        vgg19_api = VGG19("../vgg19.npy")
        vgg_map_targets = vgg19_api.build((target_224 + 1) / 2, is_rgb=(in_channels == 3))
        vgg_map_predict = vgg19_api.build((predict_224 + 1) / 2, is_rgb=(in_channels == 3))

        content_loss = tf.losses.mean_squared_error(target_imgs, pred_content)
        vgg_loss = 2e-6 * tf.losses.mean_squared_error(vgg_map_targets, vgg_map_predict)
        l1_loss = tf.reduce_mean(tf.abs(target_imgs - pred_imgs))
        mse_loss = tf.losses.mean_squared_error(target_imgs, pred_imgs)

        loss_op = content_loss + 2*vgg_loss + l1_loss

    # --------------------------------- solver definition ---------------------------------
    global_step = tf.Variable(0, name='global_step', trainable=False)
    iters_per_epoch = np.floor_divide(num, batch_size)
    lr_decay = tf.train.polynomial_decay(learning_rate=learning_rate,
                                         global_step=global_step,
                                         decay_steps=iters_per_epoch*n_epochs,
                                         end_learning_rate=learning_rate / 100.0,
                                         power=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope('optimizer'):
        with tf.control_dependencies(update_ops):
            gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("PRLNet")]
            gen_optim = tf.train.AdamOptimizer(lr_decay, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(loss_op, var_list=gen_vars)
            train_op = gen_optim.apply_gradients(gen_grads_and_vars, global_step=global_step)

    # --------------------------------- model training ---------------------------------
    '''
    if debug_mode:
        with tf.name_scope('summarise') as sum_scope:
            tf.summary.scalar('loss', loss_op)
            tf.summary.scalar('learning rate', lr_decay)
            tf.summary.image('predicts', pred_imgs, max_outputs=9)
            summary_op = tf.summary.merge_all()
    '''

    with tf.name_scope("parameter_count"):
         num_parameters = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    # set GPU resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45

    saver = tf.train.Saver(max_to_keep=1)
    loss_list = []
    psnr_list = []
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.global_variables_initializer())
        print(">>------------>>> [Training_Num] =%d" % num)
        print(">>------------>>> [Parameter_Num] =%d" % sess.run(num_parameters))

        '''
        if debug_mode:
            with tf.name_scope(sum_scope):
                summary_writer = tf.summary.FileWriter(graph_dir, graph=sess.graph)
        '''
        for epoch in range(0, n_epochs):
            start_time = time.time()
            epoch_loss, n_iters = 0, 0
            for step in range(0, num, batch_size):
                _, loss = sess.run([train_op, loss_op])
                epoch_loss += loss
                n_iters += 1
                # iteration information
                if n_iters % display_steps == 0:
                    tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    print("%s >> [%d/%d] iter: %d  loss: %4.4f" % (tm, epoch, n_epochs, n_iters, loss))
                    '''
                    if debug_mode:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                    '''

            # epoch information
            epoch_loss = epoch_loss / n_iters
            loss_list.append(epoch_loss)
            print("[*] ----- Epoch: %d/%d | Loss: %4.4f | Time-consumed: %4.3f -----" %
                  (epoch, n_epochs, epoch_loss, (time.time() - start_time)))

            if (epoch+1) % save_epochs == 0:
                if debug_mode:
                    print("----- validating model ...")
                    mean_psnr, nn = 0, 0
                    for idx in range(0, num_val, batch_size):
                        predicts, groundtruths = sess.run([pred_val, target_val])
                        save_images_from_batch(predicts, ouput_dir, idx)
                        psnr = measure_psnr(predicts, groundtruths)
                        mean_psnr += psnr
                        nn += 1
                    psnr_list.append(mean_psnr / nn)
                    print("----- psnr:%4.4f" % (mean_psnr / nn))

                print("----- saving model  ...")
                saver.save(sess, os.path.join(checkpt_dir, "model.cpkt"), global_step=global_step)
                save_list(os.path.join(ouput_dir, "loss"), loss_list)
                save_list(os.path.join(ouput_dir, "psnr"), psnr_list)

        # stop data queue
        coord.request_stop()
        coord.join(threads)
        # write out the loss list
        save_list(os.path.join(ouput_dir, "loss"), loss_list)
        save_list(os.path.join(ouput_dir, "psnr"), psnr_list)
        print("Training finished!")

    return None


def evaluate(test_list, checkpoint_dir, save_dir_test):
    print('Running PRLNet -Evaluation!')
    exists_or_mkdir(save_dir_test)
    # --------------------------------- set model ---------------------------------
    # data fetched within range: [-1,1]
    input_imgs, target_imgs, num = input_producer(test_list, in_channels, batch_size, need_shuffle=False)
    contents, details, pred_imgs = gen_PRLNet(input_imgs, out_channels, is_train=False, reuse=False)

    # --------------------------------- evaluation ---------------------------------
    # set GPU resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.45

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Restore model weights from previously saved model
        check_pt = tf.train.get_checkpoint_state(checkpoint_dir)
        if check_pt and check_pt.model_checkpoint_path:
            saver.restore(sess, check_pt.model_checkpoint_path)
            print('model is loaded successfully.')
        else:
            print('# error: loading checkpoint failed.')
            return None

        cnt = 0
        psnr_list = []
        ssim_list = []
        start_time = time.time()
        while not coord.should_stop():
            tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            print('%s evaluating: [%d - %d]' % (tm, cnt, cnt+batch_size))
            pd_images, gt_images = sess.run([pred_imgs, target_imgs])
            save_images_from_batch(pd_images, save_dir_test, cnt)
            psnr, ssim = measure_quality(pd_images, gt_images)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            cnt += batch_size
            if cnt >= num:
                coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        print("Testing finished! consumes %f sec" % (time.time() - start_time))
        print("Numerical accuracy computing ...")
        # numerical evaluation
        mean_psnr = np.mean(np.array(psnr_list))
        stde_psnr = np.std(np.array(psnr_list))
        mean_ssim = np.mean(np.array(ssim_list))
        stde_ssim = np.std(np.array(ssim_list))
        save_path = os.path.join(save_dir_test, "accuracy.txt")
        with open(save_path, 'w') as f:
            f.writelines('mean psnr:' + str(mean_psnr) + '\n')
            f.writelines('stde psnr:' + str(stde_psnr) + '\n\n')
            f.writelines('mean ssim:' + str(mean_ssim) + '\n')
            f.writelines('stde psnr:' + str(stde_ssim) + '\n')
        print("Done!")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--train_dir', type=str, default='../Data_HT/dif/train/', help='train, test')
    parser.add_argument('--val_dir', type=str, default='../Data_HT/dif/val/', help='train, test')
    parser.add_argument('--test_dir', type=str, default='../Data_HT/dif/test/', help='train, test')    
    parser.add_argument('--output_dir', type=str, default='./output', help='train, test')
    args = parser.parse_args()

    if args.mode == 'train':
        train_list = gen_list(args.train_dir)
        val_list = gen_list(args.val_dir)
        train(train_list, val_list, debug_mode=True)
    elif args.mode == 'test':
        test_list = gen_list(args.test_dir)
        checkpoint_dir = "checkpoints"
        evaluate(test_list, checkpoint_dir, args.output_dir)
    else:
        raise Exception("Unknow --mode")

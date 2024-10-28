import json
from matplotlib import pyplot as plt
import torch
import numpy as np
import argparse
import os
import sys
import time
import random
import datetime
from soft_ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, pkl_save, data_dropout
from scipy.signal import find_peaks
import pmdarima as pm
from pmdarima.model_selection import train_test_split

from utils_distance_matrix import * 
import sys

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

def fix_seed(expid):
    SEED = 2000 + expid
    random.seed(SEED)
    np.random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--dist_type', type=str, default='DTW')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--tau_inst', type=float, default=0)
    parser.add_argument('--tau_temp', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lambda_', type=float, default=0.5)
    parser.add_argument('--expid', type=int, default=2)
    parser.add_argument('--separate_reg', action="store_true", help='Whether to perform weighting in temporal loss')
    parser.add_argument('--use_arima', action="store_true", help='use arima to find parameters')
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    soft_instance = (args.tau_inst > 0)
    soft_temporal = (args.tau_temp > 0)
    soft_cl = soft_instance + soft_temporal

    if 'forecast' in args.loader:
        soft_instance = False
    
    if args.loader in ['UCR','UEA']:
        result_name = f'results_classification_{args.loader}/'
    else:
        result_name = f'results_{args.loader}/'
    
    
    if args.use_arima:
        run_dir = result_name +f"arima_INST{int(soft_instance)}"
        run_dir += f'_tau_temp_auto_tau_inst{float(args.tau_inst)}' 
        soft_temporal = True
    else:
        run_dir = result_name + f'TEMP{int(soft_temporal)}_INST{int(soft_instance)}'
        if soft_cl:
            run_dir += f'_tau_temp{float(args.tau_temp)}_tau_inst{float(args.tau_inst)}' 
    
    if args.lambda_ == 0:
        run_dir += '_no_instCL'
        
    run_dir = os.path.join(run_dir, args.dataset, f'bs{args.batch_size}', f'run{args.expid}')
    os.makedirs(run_dir, exist_ok=True)
    
    file_exists = os.path.join(run_dir,f'eval_res_{args.dist_type}.pkl')
        
    if os.path.isfile(file_exists):
        print('You alreay have the results. Bye Bye~')
        sys.exit(0)
    
    fix_seed(args.expid)
    device = init_dl_program(args.gpu, seed=args.seed)#, max_threads=args.max_threads)

    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels, sim_mat = datautils.load_UCR(args.dataset, args.dist_type)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels, sim_mat = datautils.load_UEA(args.dataset, args.max_train_length)
    
    elif args.loader == 'semi':
        task_type = 'semi-classification'
        train_data, train_labels, train1_data, train1_labels, train5_data, train5_labels, test_data, test_labels, sim_mat = datautils.load_semi_SSL(args.dataset)
    
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        UNI = False
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, args.max_train_length, univar=UNI)
        train_data = data[:, train_slice]
        print("shape:",train_data.shape)
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        UNI = True
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, args.max_train_length, univar=UNI)
        train_data = data[:, train_slice]
        print("shape:",train_data.shape)
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        train_data, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset, args.max_train_length, cold=False)
        print("shape:",train_data.shape)

    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        _, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset, args.max_train_length, cold=True)
        train_data, _, _, _, _ = datautils.load_UCR('FordA')
        print("shape:",train_data.shape)

    else:
        raise ValueError(f"Unknown loader {args.loader}.")
    
    #returns top seasons in ascending order
    def getSeasons(data, num_of_top_s = 3, threshold = 5, print_fft = False):
        fft = np.abs(np.fft.fft(np.diff(data,append=[0])))
        
        fft = fft[:len(data) // 2]
        peak_indices, _ = find_peaks(fft, distance= 3)
        _mean_median_th = max([fft.mean()*threshold, np.median(fft)*threshold])
        peak_indices = peak_indices[fft[peak_indices] > _mean_median_th ]
        if  len(peak_indices)==0:
            return None
        num_of_top_s = len(peak_indices) if len(peak_indices)<num_of_top_s else num_of_top_s
        top_peaks = peak_indices[np.argsort(fft[peak_indices])][-num_of_top_s:]
        top_seasons = np.rint(len(data)/top_peaks).astype(int)

        print("top_seasons: ",top_seasons, "\n tot peaks num:", len(peak_indices))
        if print_fft:
            plt.axhline(y = _mean_median_th, color = 'r', linestyle = 'dashed')    
            plt.plot(fft)
            plt.plot(peak_indices, fft[peak_indices], 'o')
            plt.show()
        return top_seasons
    
    #iterate over the features 
    if args.use_arima and task_type == 'forecasting' and UNI == True:
        temp_data = train_data.transpose(2, 0, 1)[-1][0]
        seasons = getSeasons(temp_data, print_fft = True)
        season = 1 if seasons is None else seasons[-1]

        #plt.plot(train_data[0])
        #plt.show()
        model = pm.auto_arima(temp_data[:len(temp_data)//8], 
                              seasonal=True,
                              #max_p=None,
                              #max_q=None,
                              #max_P=None,
                              #max_Q=None,
                              max_order=None,
                              m=season
                              )
        print(model.summary(),"\n",model.order,model.seasonal_order, model.get_params().get('seasonal_order'))
        tau_temp = sum(model.order[0:2])+sum(model.seasonal_order[0:2])*season

    elif args.use_arima:
        orders_sum = []
        for f_idx, feature in enumerate(train_data.transpose(2, 0, 1)):
            for inst_idx, inst in enumerate(feature):
                seasons = getSeasons(inst)
                season = 1 if seasons is None else seasons[-1]
                model = pm.auto_arima(train_data.transpose(2, 0, 1)[-1][0], 
                                    seasonal=True,
                                    #max_p=None,
                                    #max_q=None,
                                    #max_P=None,
                                    #max_Q=None,
                                    max_order=None,
                                    m=season
                                    )
                print(f"order of the feature #{f_idx}, and instance {inst_idx} - ", model.order)
                orders_sum.append(sum(model.order[0:2])+sum(model.seasonal_order[0:2])*season)

        tau_temp = 2./max(orders_sum)
        print ("tau temporal:", tau_temp)
    else:
        tau_temp = args.tau_temp


    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    #args.tau_temp = 2.0
    #print("Arguments:", str(args))

    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        tau_temp=tau_temp,
        lambda_ = args.lambda_,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        soft_instance = soft_instance,
        soft_temporal = soft_temporal
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
    
    t = time.time()

    model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )

    if soft_instance:
        dist_mat = 1 - sim_mat
        del sim_mat
        sim_mat = densify(-dist_mat, args.tau_inst, args.alpha)
        print('Soft Assignment Matrix', sim_mat.shape)
        print('sudgestes tau')
    
    if task_type in ['forecasting','anomaly_detection','anomaly_detection_coldstart']:
        sim_mat = None
        
    if task_type == 'classification':
        loss_log = model.fit(
            train_data, train_labels, test_data, test_labels,
            sim_mat, run_dir, n_epochs=args.epochs, n_iters=args.iters,
            verbose=True)
    else:
        loss_log = model.fit(
            train_data, 0, 0, 0,
            sim_mat,run_dir, n_epochs=args.epochs, n_iters=args.iters,
            verbose=True)
    
    #model.save(f'{run_dir}/model.pkl')
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            if args.dataset == 'electricity':
                if args.separate_reg:
                    out, eval_res = tasks.eval_forecasting_separate(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar=True)
                else:
                    out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar=True)
            else:
                if args.separate_reg:
                    out, eval_res = tasks.eval_forecasting_separate(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar=True)
                else:
                    out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols,univar=UNI)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'semi-classification':
            out, eval_res = tasks.eval_semi_classification(model, 
                                                           train1_data, train1_labels, 
                                                           train5_data, train5_labels, 
                                                           train_labels,
                                                           test_data, test_labels, eval_protocol='svm')
        else:
            assert False
            
        #pkl_save(f'{run_dir}/out.pkl', out)
        #pkl_save(f'{run_dir}/eval_res_{args.dist_type}.pkl', eval_res)
        print('Evaluation result:', json.dumps(eval_res, indent=4))
        if task_type == 'classification':
            fname = "classification.csv"
            header = ""
            if not os.path.isfile(fname):
                header="loader\tdataset\tauto_s\ttau_temp\ttau_inst\tlambda\tacc\tauprc"
            with open(fname, "a") as file:
                use_auto= "y" if args.use_arima else "n"
                line = header+f"\n{args.loader}\t{args.dataset}\t{use_auto}\t{tau_temp}\t{args.tau_inst}\t{args.lambda_}\t"
                line +="\t".join([str(eval_res[i]) for i in ["acc","auprc"]])
                file.write(line)
            
        if args.loader == 'forecast_csv_univar':
            fname = "forecast_csv_univar.csv"
            header = ""
            if not os.path.isfile(fname):
                cols = []
                for size in [24,48,168,336,720]:
                    suffix = "-" + str(size)
                    for i in ["nr","rw"]:
                        cols.append("MSE-"+i+suffix)
                        cols.append("MAE-"+i+suffix)

                header="loader\tdataset\tauto_s\ttau_temp\ttau_inst\tlambda\t"+"\t".join(cols)
            with open(fname, "a") as file:
                use_auto= "y" if args.use_arima else "n"
                line = header+f"\n{args.loader}\t{args.dataset}\t{use_auto}\t{tau_temp}\t{args.tau_inst}\t{args.lambda_}\t"
                vals = []
                for size in [24,48,168,336,720]:
                    for tp in ["norm","raw"]:
                        vals.append(str(eval_res["ours"][size][tp]["MSE"]))
                        vals.append(str(eval_res["ours"][size][tp]["MAE"]))
                
                line +="\t".join(vals)
                file.write(line)            


    print("Finished.")

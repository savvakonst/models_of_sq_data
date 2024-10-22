::python -u train.py ETTh1 --loader forecast_csv_univar --repr-dims 320  --seed 42 --eval
for %%x in (5.0 10) do (
     python -u train.py ETTh1 --loader forecast_csv_univar --repr-dims 320  --seed 42 --eval --use_arima  --tau_inst %%x
     python -u train.py ETTh1 --loader forecast_csv_univar --repr-dims 320  --seed 42 --eval --tau_temp 0.2 --tau_inst %%x
     python -u train.py ETTh1 --loader forecast_csv_univar --repr-dims 320  --seed 42 --eval --tau_temp 0.5 --tau_inst %%x
     python -u train.py ETTh1 --loader forecast_csv_univar --repr-dims 320  --seed 42 --eval --tau_temp 1.0 --tau_inst %%x
     python -u train.py ETTh1 --loader forecast_csv_univar --repr-dims 320  --seed 42 --eval --tau_temp 2.0 --tau_inst %%x
     python -u train.py ETTh1 --loader forecast_csv_univar --repr-dims 320  --seed 42 --eval --tau_temp 4.0 --tau_inst %%x
)
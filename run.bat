for %%x in ( DiatomSizeReduction) do (
     python -u train.py %%x  --loader UCR --batch-size 8 --repr-dims 320  --seed 42 --eval --use_arima
)
     ::python -u train.py %%x  --loader UCR --batch-size 8 --repr-dims 320  --seed 42 --eval 
     ::python -u train.py %%x  --loader UCR --batch-size 8 --repr-dims 320  --seed 42 --eval --tau_temp 0.2
     ::python -u train.py %%x  --loader UCR --batch-size 8 --repr-dims 320  --seed 42 --eval --tau_temp 0.5
     ::python -u train.py %%x  --loader UCR --batch-size 8 --repr-dims 320  --seed 42 --eval --tau_temp 1.0
     ::python -u train.py %%x  --loader UCR --batch-size 8 --repr-dims 320  --seed 42 --eval --tau_temp 2.0
     ::python -u train.py %%x  --loader UCR --batch-size 8 --repr-dims 320  --seed 42 --eval --tau_temp 4.0

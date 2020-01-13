virtualenv -p python3 cpu_env
. cpu_env/bin/activate
pip install -r requirements_cpu.txt
mkdir data
mkdir models
wget -N pollithy.com/rnn_model_fake_data.h5
mv rnn_model_fake_data.h5 models/rnn_model_fake_data.h5
python get_data.py
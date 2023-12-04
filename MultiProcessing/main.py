from functions import seventeenTrafficLights

def main(accumulated_epoc: int, epoc_this_run: int):
	load_model_name = f"./models/PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{accumulated_epoc}e8[-2,2]"
	trained_model_name = f"./models/PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{accumulated_epoc + epoc_this_run}e8[-2,2]"
	training_iterations = epoc_this_run * 1e8
	seventeenTrafficLights(load_model_name, trained_model_name, training_iterations)

main(accumulated_epoc=380, epoc_this_run=20)                                      


# .tb_log/1020 PPO_13 PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_380e8[-2,2]"
# .tb_log/1020 PPO_14 PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_400e8[-2,2]"


# python3 -m tensorboard.main --logdir=./tb_log/1020 --port=6006
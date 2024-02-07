from functions import seventeenTrafficLightsSettings

def main(accumulated_epoc: int, epoc_this_run: int):
	env_id = "SeventeenTrafficLights-v2"
	load_model_name = f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_{accumulated_epoc}e8[-2,2]"
	trained_model_name = f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_{accumulated_epoc + epoc_this_run}e8[-2,2]"
	training_iterations = epoc_this_run * 1e8
	seventeenTrafficLightsSettings(env_id, load_model_name, trained_model_name, training_iterations)

# main(accumulated_epoc=340, epoc_this_run=20)

def checkModel():
	from check_model import check_model

	check_model(env_name="SeventeenTrafficLights-v2", 
				model_name="./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_360e8[-2,2]",
				output_file_name="20240109.csv",
				header=None,
				output_path="/home/carla/Owen/"
				)

checkModel()


# ./tb_log/1020/PPO10 f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_320e8[-2,2]
# ./tb_log/1201/PPO 2 f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_360e8[-2,2] UNFINISHED DUE TO POWER OUTAGE
# ./tb_log/1201/PPO10 f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_340e8[-2,2]
# ./tb_log/1201/PPO19 f"./models/PPO_SeventeenTrafficLightsV2_2048_3e-5_deltat_0.1_360e8[-2,2]
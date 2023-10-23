from functions import seventeenTrafficLights

def main():
	load_model_name = f"./models/PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{200}e8[-2,2]"
	trained_model_name = f"./models/PPO_SeventeenTrafficLights_2048_3e-5_deltat_0.1_{220}e8[-2,2]"
	training_iterations = 20e8
	seventeenTrafficLights(load_model_name, trained_model_name, training_iterations)

main()
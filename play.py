from Games.ConnectFour import ConnectFourPolicyEstimator, ConnectFour
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

work_dir = "."
#work_dir = os.environ["WORK"]

checkpoint_location = work_dir + "/checkpoints/checkpoint_net_connect4.json"

pe = ConnectFourPolicyEstimator()
pe.load(checkpoint_location)
game = ConnectFour

game.test_policy_human(pe, ai_player=0, samples=20, batch_size=40, discount=0.997)


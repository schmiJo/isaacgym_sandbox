
from tasks.cartpole.cartpole_task import Cartpole
import torch

if __name__ == "__main__":
    device = "cuda:0"
 
    
    env = Cartpole({}, "cuda:0", 0, False)
    
    env.is_symmetric = False
    
    env.reset()
    
    while True:
    
        
        action = torch.tensor([ env.action_space.sample() for _ in range(env.num_envs)])
        # command 
        # state 

        #print(action)


        #action = np.ones((num_agents, env.get_action_size()))
        obss, rewards, dones, _ =   env.step(action)

        print("actor")
        print(obss[0]["linear"].shape)
        print("critic")
        print(obss[1]["linear"].shape)
        
        
        
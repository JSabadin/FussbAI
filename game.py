import requests
import json
import time
import numpy as np
from itertools import chain

HOST_ADDRESS = '127.0.0.1:23336'

def get_camera_data():
    """
    Fetches the current state of the camera from the game server.
    
    Returns:
        dict: A dictionary containing the current camera data.
    """
    cam_url = f"http://{HOST_ADDRESS}/Camera/State"
    response = requests.get(cam_url)    
    return response.json()


def get_competition_state():
    """
    Fetches the current state of the competition from the game server.
    
    Returns:
        dict: A dictionary containing the current competition data.
    """
    cam_url = f"http://{HOST_ADDRESS}/Competition"
    response = requests.get(cam_url)    
    return response.json()

def send_motor_commands(cmds):
    """
    Sends motor commands to the game server.
    
    Args:
        cmds (dict): A dictionary of commands to send to the motor controller.
    """
    motors_url = f"http://{HOST_ADDRESS}/Motors/SendCommand?blue=False"
    response = requests.post(motors_url, json=cmds)

class SoccerEnv:
    def __init__(self):
        """
        Initializes the Soccer environment, loading field geometry from a JSON file and setting initial scores and flags.
        Initialize environment state and parameters
        Load field geometry from json  
        Geometry contents:
        - field: field dimensions (in mm)
        - rods: array of player rod information
                * id: 1 to 8 (1 is the red goalie)
                * team: red or blue
                * position: x-axis position of the rod
                * travel: travel range (in mm) of the rod
                * players: number of players on the rod
                * first_offset: y-axis position of the first player center
                * spacing: spacing between players on the rod
        """
 
        f = open('geometry.json')
        self.geometry = json.load(f)
        f.close()
        self.rod_num = 4 # Number of rods we are controlling
        self.goal_flag = False  # Initialize the goal fl
        self.player_score = 0
        self.enemy_score = 0
        self.previous_ball_x_vel = 0
        self.previous_ball_y_vel = 0

    def action_to_motor(self, action):
        """
        Converts action list to motor commands and sends them to the game server.
        
        Args:
            action (list): A list of actions to be converted to motor commands.
        """
        for rod_idx in range(self.rod_num):
            motorData = []
            # Adjusted to slice 4 elements per action
            pos_rot, w_rot, pos_rod, v_rod = action[rod_idx * 4: (rod_idx+1) * 4] # action is a list of 4 motor commands for each rod!
            cmd = {
                "driveID": rod_idx,                       # Enemy Rod index 2,4,6,7 -> 3 forwards, 5 midfielders, 2 defenders, 1 goalies || Friendly Rod index 0,1,3,5 -> 1 goalies, 2 defenders, 5 midfielders, 3 forwards
                "rotationTargetPosition": pos_rot,        # angle of rod     [-1,1]
                "rotationVelocity": w_rot,                # angle velocity    [0,1]
                "translationTargetPosition": pos_rod,     # rod position      [0,1]
                "translationVelocity": v_rod              # translational speed  [0,1]
            }        
            motorData.append(cmd)
            send_motor_commands({'commands': motorData})

    def reset(self):
        """
        Resets the environment to its initial state and sends default motor commands to ensure all rods are in their default positions.
        """
        # Call this at the start of each episode
        action = []
        for i in range(16):
            if i%4 in [1, 3]:
                action.append(1)
            else:
                action.append(0)
        self.action_to_motor(action)
        # Reset the score to 0:0
        self.player_score = 0
        self.enemy_score = 0
        camData = get_camera_data() 
        # Get the state
        state, _ ,_ = self.get_state_done_reward(camData)

        return state

    def ball_check(self):
        """
        Fetches the information from the game server about the ball being on the valid region of the table.
        
        Returns:
            Bool: The flag for the valid ball.
        """
        camData = get_camera_data() 
        CD0 = camData["camData"][0]
        CD1 = camData["camData"][1]      
        
        # Process camera data
        ball_x_pose = (CD0["ball_x"] + CD1["ball_x"]) / 2
        ball_y_pose = (CD0["ball_y"] + CD1["ball_y"]) / 2
        goal_offset = 20 # For the goal region
        if(ball_y_pose > 0 and ball_y_pose < self.geometry["field"]["dimension_y"]) and (ball_x_pose > 0 - goal_offset) and (ball_x_pose < self.geometry["field"]["dimension_x"] + goal_offset):
            return True
        else :
            return False


    def get_state_done_reward(self, camData):
        """
        Fetches the state, reward, and done flag based on the camera data and the current game state.
        """
        # Initialize variables
        done = False
        reward = 0

        playerMapping = [1, 2, -1, 3, -1, 4, -1, -1] 
        CD0 = camData["camData"][0]
        CD1 = camData["camData"][1]      

        # Process camera data
        ball_x_pose = (CD0["ball_x"] + CD1["ball_x"]) / 2
        ball_y_pose = (CD0["ball_y"] + CD1["ball_y"]) / 2
 
        # Average ball velocity
        ball_x_vel = (CD0["ball_vx"] + CD1["ball_vx"]) / 2
        ball_y_vel = (CD0["ball_vy"] + CD1["ball_vy"]) / 2

        # Average ball size
        ball_size =  (CD0["ball_size"] + CD1["ball_size"]) / 2
  
        # PLAYER POSITIONS
        relative_friendly_y_positions = [] # 1 goalies, 2 defenders, 5 midfielders, 3 forwards
        friendly_players_rod_rotations = [] # 1 goalies, 2 defenders, 5 midfielders, 3 forwards, but the data is we are receiving is in order 3,5,2,1 !

        relative_enemy_y_positions = [] # 1 goalies, 2 defenders, 5 midfielders, 3 forwards BUT we are receiving them in order 3,5,2,1 !
        enemy_players_rod_rotations = [] # 1 goalies, 2 defenders, 5 midfielders, 3 forwards, but the data is we are receiving is in order 3,5,2,1 !

        for i in range(8):  # Loop through eight rods
            rods_info = self.geometry["rods"][i]
            player_x_position = rods_info["position"]
            num_of_players_in_rod = rods_info["players"]
            for player_idx in range(num_of_players_in_rod):
                # Calculate position
                pos = (rods_info["travel"] * CD0["rod_position_calib"][i] + rods_info["travel"] * CD1["rod_position_calib"][i])/2
                player_position_y = rods_info["first_offset"] + pos + player_idx * rods_info["spacing"]
                
                # Append position based on mapping
                if playerMapping[i] < 0: # Enemy rod
                    relative_enemy_y_positions .insert(0, (player_position_y - ball_y_pose)/self.geometry["field"]["dimension_x"])  # Insert at beginning to reverse order 3,5,2,1 -> 1,2,5,3
                else: # Friendly rod
                    relative_friendly_y_positions .append((player_position_y - ball_y_pose)/self.geometry["field"]["dimension_x"])
                    # Reward for being close to the ball
                    dx = np.abs(player_x_position - ball_x_pose)
                    dy = np.abs(player_position_y - ball_y_pose)
                    if dy < 30 and dx < 50: 
                        reward += 0.3  # Reward for being close to the ball
                        if self.previous_ball_x_vel * ball_x_vel < 0 and self.previous_ball_x_vel < 0:
                            reward += 7  # Reward for a successful block
                            # print("Block Successful")
                        
            if playerMapping[i] < 0:
                enemy_players_rod_rotations.insert(0, (CD0["rod_angle"][i]+CD1["rod_angle"][i]) / 2 / 32)  # Insert at beginning to reverse order 3,5,2,1 -> 1,2,5,3
            else:
                friendly_players_rod_rotations.append((CD0["rod_angle"][i]+CD1["rod_angle"][i]) / 2 / 32)


            # Predict future ball positions (for simplicity, predict one step ahead)
        delta_t = 0.1  # Future time step to predict (you can adjust this based on your simulation's time step)
        future_ball_x_pose = (ball_x_pose + ball_x_vel * delta_t)/self.geometry["field"]["dimension_x"]
        future_ball_y_pose = (ball_y_pose + ball_y_vel * delta_t)/self.geometry["field"]["dimension_y"]

        state = list(chain([ball_x_pose/self.geometry["field"]["dimension_x"], ball_y_pose/self.geometry["field"]["dimension_x"], ball_x_vel, ball_y_vel, future_ball_x_pose, future_ball_y_pose], relative_friendly_y_positions, friendly_players_rod_rotations, relative_enemy_y_positions, enemy_players_rod_rotations))

        # Reward calculation, where the goal logic is implemented
        goal_scored = goal_received = False
        goal_width = 200
        # Check for a goal scored or received only if the goal flag is not set
        if (not self.goal_flag) and (self.geometry["field"]["dimension_y"] / 2 - goal_width/2 < ball_y_pose < self.geometry["field"]["dimension_y"] / 2 + goal_width/2):
            if ball_x_pose > self.geometry["field"]["dimension_x"]:
                goal_scored = True
                self.player_score += 1  # Increment player's score
                self.goal_flag = True  # Set the flag indicating a goal event has occurred
            elif ball_x_pose < 0:
                self.enemy_score += 1  # Increment enemy's score
                goal_received = True
                self.goal_flag = True  # Set the flag indicating a goal event has occurred

        # Reset the goal flag based on a condition, such as the ball returning to a neutral area
        # For example, if the ball is reset to the middle of the field after a goal
        if self.goal_flag and (0 < ball_x_pose < self.geometry["field"]["dimension_x"]):
            self.goal_flag = False  # Reset the flag once the ball is back in play

        if self.geometry["field"]["dimension_x"] > ball_x_pose > self.geometry["field"]["dimension_x"] / 2:
            in_attacking_half = True
            in_defensive_half = False
        elif self.geometry["field"]["dimension_x"] < ball_x_pose < self.geometry["field"]["dimension_x"] / 2:
            in_defensive_half = True
            in_attacking_half = False
        else:
            in_attacking_half = in_defensive_half = False

        # Assign rewards based on ball position
        reward += 30 if goal_scored else -30 if goal_received else 1e-2 if in_attacking_half else -1e-2 if in_defensive_half else 0

        # Update the ball velocity tracking
        self.previous_ball_x_vel = ball_x_vel
        self.previous_ball_y_vel = ball_y_vel

        # Check if either player or enemy has reached 3 goals
        if self.player_score >= 3 or self.enemy_score >= 3:
            done = True
        else:
            done = False

        return state, reward, done
        

    def step(self, action):
        """
        Executes a step in the environment using the provided action, updates the game state, and returns the new state, reward, and done flag.
        
        Args:
            action (list): The action to execute.
            
        Returns:
            tuple: Contains the new state (list), reward (int), done flag (bool), and additional info (dict).
        """
   
        # Logic to send motor commands to the simulator
        self.action_to_motor(action)
  
        # Read the camera data and return the states
        camData = get_camera_data() 

        # Get the state, reward, and done flag
        state, reward, done = self.get_state_done_reward(camData)
      
        
        return state, reward, done



if __name__ == "__main__":
    # Initialize the environment
    env = SoccerEnv()

    # Create a mock action for testing; adjust based on your action format
    # Example action: [rod_idx, pos_rot, w_rot, pos_rod, v_rod] for each of the 4 rods
    while True:
        time.sleep(0.06)
        # action = []
        # # Random noise for testing
        # for i in range(16):
        #         action.append(random.uniform(1, -1) if i % 4 == 0 else random.uniform(0, 1))
            
        action = []
        for i in range(16):
            if i%4 in [1, 3]:
                action.append(1)
            else:
                action.append(0)

        if env.ball_check(): # Ball is inside the table or in the goal area
            state, reward, done = env.step(action)
  
        if reward == -100 or reward == 100:
            print("Reward:", reward)

        if done:
            env.reset()
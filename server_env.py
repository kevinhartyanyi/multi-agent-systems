import gym
from gym import spaces, logger
import numpy as np
import itertools

"""Example: 
        {'lastActionParams': [], 'score': 0, 'lastAction': '', 'things': [{'x': 0, 'y': 0, 'details': 
        'A', 'type': 'entity'}], 'attached': [], 'disabled': False, 'terrain': {'goal': [[0, 4], [-1, 4], [0, 5]]}, 
        'lastActionResult': '', 'tasks': [], 'energy': 300} """

"""{'lastActionParams': ['w'], 'score': 0, 'lastAction': 'move', 'things': [{'x': 0, 'y': 0, 'details': 'A', 
'type': 'entity'}], 'attached': [], 'disabled': False, 'terrain': {'obstacle': [[3, 2], [2, 2], [1, 2], [0, 2], [-1, 
2], [-2, 2], [-5, 0], [-3, 2], [1, -4], [0, -4], [-1, -4]]}, 'lastActionResult': 'success', 'tasks': [{'reward': 10, 
'requirements': [{'x': 0, 'y': 1, 'details': '', 'type': 'b2'}], 'name': 'task11', 'deadline': 408}, """

"""
'tasks': [{'reward': 10, 
'requirements': [{'x': 0, 'y': 1, 'details': '', 'type': 'b1'}], 'name': 'task1', 'deadline': 203}, 
{'reward': 10, 'requirements': [{'x': 0, 'y': 1, 'details': '', 'type': 'b2'}], 'name': 'task0', 'deadline': 126}]
"""

"""
"clearEnergyCost" : 50
"disableDuration" : 4,
"maxEnergy" : 300,
"attachLimit" : 10

"grid" : {
        "height" : 40,
        "width" : 40,
        "file" : "conf/maps/test40x40.bmp"
      },


"blockTypes" : [3, 3],
"dispensers" : [2, 3]

"tasks" : {
        "size" : [1, 1],
        "duration" : [100, 200],
        "probability" : 0.05
      }
"""


def vision_grid_size(vision):
    """
    Calculates the maximum visible cell amount based on vision.
    """
    if vision > 0:
        return 4 * vision + vision_grid_size(vision - 1)
    else:
        return 0


def convert_indexes_to_observation_vector(x, y, vision, vision_grid_size):
    def current_step(size, steps):
        if steps <= 0:
            return 0
        step = size - 2
        return step + (current_step(step, steps - 1))

    base = vision_grid_size / 2
    base_width = 2 * vision + 1

    base_add = (1 if x != 0 else 0) * abs(x)

    ind = -1
    if x > 0:
        ind = (base - (base_add + current_step(base_width, abs(x)))) - y
    else:
        ind = (base + (base_add + current_step(base_width, abs(x)))) - y

    return int(ind)


def vision_array(vision):
    re = []
    for v in range(-vision, vision + 1):
        re += itertools.product([v], list(range((-vision) + abs(v), (vision + 1) - abs(v))))

    return [list(l) + [0,0] for l in re]

def find_ind_in_observation_vector(vector, val):
    ind = -1
    for i,v in enumerate(vector):
        if v[:2] == val:
            ind = i
    return ind

def find_ind_in_observation_np_array(array, val):
    ind = -1
    for i,v in enumerate(array):
        x,y = v[:2]
        if x == val[0] and y == val[1]:
            ind = i
    return ind

def dispenser_convert_to_int(dispenser_name):
    re = -1
    if dispenser_name == "b0": # 1 is entity A
        re = 2
    elif dispenser_name == "b1":
        re = 3
    elif dispenser_name == "b2":
        re = 4
    return re

task_dict = {}


class Server(gym.Env):
    """
    Description:
        Handles the communication with the server and manages the environment for the agents.

    Actions:
        Type: Discrete(4) TODO: Implement the other actions
        Num   Action
        0     move north
        1     move east
        2     move south
        3     move west

    Observation:
        Type: Box(vision_size, 2)
        Num             Dim        Observation               Min                     Max
        0               0           Cell Type                 0                       3
        0               1           Thing Type                0                       4
        1               --          --                        --                      --
        ...             --          --                        --                      --
        vision_size     --          --                        --                      --

        Cell Types:
            Num   Type
            0     Empty
            1     Obstacle
            2     Goal

        Thing Types:
            Num   Type
            0     Nothing
            1     Entity
            2     Block
            3     Dispenser
            4     Marker

    Reward:
        Currently using the 'score' returned by the server in every request-action message.
        TODO: More robust reward implementation

    Starting State:
        All observations are from the server

    Episode Termination:
        Server returns 'sim-end' message.
        TODO: More stopping points (maybe)

    """

    def __init__(self):
        #self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.connect()

        """Agents perceive the state of a cell depending on their vision. E.g. if they have a vision of 5, 
        they can sense all cells that are up to 5 steps away. """
        self.agent_vision = 5 ##TODO

        self.action_space = spaces.Discrete(4) # NOT INCLUSIVE

        self.vision_size = vision_grid_size(5)

        low = np.zeros((self.vision_size, 4))
        high = np.zeros((self.vision_size, 4))
        low[:, :2] = -self.agent_vision
        high[:, :2] = self.agent_vision
        high[:, 2:3] = 2  # cell type -> high is inclusive
        high[:, 3:] = 4  # thing type

        # Other alternative: spaces.Dict()
        """
        shape = (x-1, 2)
            x is the amount of cells that agent can see
            the second dimension contains information about the cell:
                - [0] the type of the cell (empty if not specified)
                - [1] the thing on the cell (if any)
            low,high: Independent bounds for each dimension
        """
        """ Observation_space
            - lastActionParams: Discrete(4)
            - lastAction: Discrete(1) ##TODO
            - lastActionResult: Discrete(2) # Failed or not
            - things/terrain: Box((vision_size,2)) ##TODO no details included
            * score
            * attached
            * energy
            * task
        """
        self.observation_space = spaces.Tuple((
            spaces.Discrete(4),
            spaces.Discrete(1),
            spaces.Discrete(2),
            spaces.Box(
            low=low,
            high=high,
            shape=((self.vision_size, 4)),
            dtype=np.int
            )
        ))

        # self.seed()
        # self.viewer = None
        self.state = None # Observation_space instance
        

        #self.steps_beyond_done = None

    def seed(self, seed=None):
        pass

    def reset(self):
        ## TODO Start server (subprocess)
        return None

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        
        reward = action.eval(self.state)
        done = False
        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        pass
        
    def update(self, msg):
        #self.state = self.observation_space.sample()

        observation_map = np.zeros((self.vision_size, 2))
        observation_map = vision_array(self.agent_vision)

        things = msg['things']
        terrain = msg['terrain']
        #print(f"\n\n\nThings: {things}")
        #print(f"Terrain: {terrain}")

        for th in things:
            x = th["x"]
            y = th["y"]
            detail = th["details"]
            typ = th["type"]

            ind = find_ind_in_observation_vector(observation_map, [x,y])

            #print(f"Thing detail: {detail}")
            #print(f"Map ind: {ind}")
            if detail == "A":
                observation_map[ind][2] = 1

            if typ == "dispenser":
                observation_map[ind][2] = dispenser_convert_to_int(detail)

        terrain_values = [("goal", 1),("obstacle", 2)]

        for tr in terrain_values:
            name, value = tr
            terran_cords = []
            try:
                terran_cords = terrain[name]
                #print(f"Terrain cords: {terran_cords}")
                for cords in terran_cords:
                    x, y = cords
                    ind = find_ind_in_observation_vector(observation_map, [x,y])
                    observation_map[ind][3] = value
            except:
                print(f"Terrain: {name} not found")


        """for ob in observation_map:
            print(ob)
        print("\n\n\n")"""

        self.state = (0,0,0,observation_map)
        #print(self.state)

        # Extract tasks
        tasks = msg["tasks"]
        preprocessed_tasks = []

        for t in tasks:
            #print("Requirements size: ", len(t["requirements"]))
            #if len(t["requirements"]) > 1:
            #    input()
            reward = t["reward"]
            requirements = t["requirements"][0] # TODO: Only using the first requirement
            name = t["name"] # TODO: Currently not used
            deadline = t["deadline"]
            details = requirements["details"] # TODO: Find a use for this
            x = requirements["x"]
            y = requirements["y"]
            dispenser = requirements["type"]

            str_name = name # Only used for visualization

            # Convert/Register task
            if name in task_dict:
                name = task_dict[name]
            else:
                counter = 0
                if len(task_dict) != 0:
                    counter = max([value for key, value in task_dict.items()]) + 1

                task_dict[name] = counter
                name = counter

            # Convert dispenser
            dispenser = dispenser_convert_to_int(dispenser)

            preprocessed_tasks.append([
                str_name, x, y, dispenser, deadline, reward
            ])

        print("Tasks: ", preprocessed_tasks)


        return self.state, preprocessed_tasks
        
"""
    def connect(self, host: str = "127.0.0.1", port: int = 12300):
        connected = False
        wait_sec = 2
        while not connected:
            try:
                self.sock.connect((host, port))
            except ConnectionRefusedError:
                print(f"Connection Refused. Trying again after {wait_sec} seconds")
                time.sleep(wait_sec)
            else:
                connected = True

    def init_agent(self, agent_message: AuthRequest):
        msg = agent_message.msg()
        print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())
        response = json.loads(self.sock.recv(4096).decode("ascii").rstrip('\x00'))
        print(f"Response: {response}")
        return True if response["content"]["result"] == "ok" else False

    def send(self, agent_message: ActionReply):
        msg = agent_message.msg()
        print(f"Sending: {msg}")
        self.sock.sendall(msg.encode())

    def receive(self):
        while True:
            recv = self.sock.recv(4096).decode("ascii").rstrip('\x00')
            if recv != "":
                break
        response = json.loads(recv.rstrip('\x00'))
        print(f"Response: {response}")
        return response
"""

my_serv = Server()
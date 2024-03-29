"""
    ### State codes
        
        ## -1 UNKNOWN
        
        ## Things
            0. - empty
            1. - entity (2)
                0. - A
                1. - B
            2. - block (block_num)
                0. - b0
                1. - b1
                2. - b2
            3. - dispenser (block_num) # dispenser_num number of dispensers, but their types can be block_num
                0. - b0
                1. - b1
                2. - b2                
            4. - marker (3?)
                0. - clear
                1. - ci ('clear immediately')
                2. - cp ('clear perimeter')
                
        ## Terrain
            0. - empty
            1. - goal
            2. - obstacle
            
        ## Tasks (task_num)
            We store a predefined number of tasks in the environment. At every iteration we check if the task
            is still available from the agent's perception. We store the names and associate them with indices.
            Once a task is due for submission, the agent can access this bookkeeping within the env and get
            the name so it can submit.
            - deadline (0-STEP_NUM)
            - reward (0-100) # no reason for this to be 100
            - requirements (TASK_SIZE x 3)
                - x
                - y
                - BLOCK_NUM
            
    ### Things we assume we know
        - VISION_RANGE (5) -> the range of vision of our agent
        - STEP_NUM (500) -> the number of steps in the game
        - MAX_ENERGY (300) -> maximum energy of the agent
        - TASK_SIZE (1) -> the maximum number of blocks used in a task
        - BLOCK_NUM (3) -> maximum number of different types of blocks
        - DISPENSER_NUM (3) -> maximum number of dispensers
        
    ### Network inputs -> Probably DQN
        # Normalize inputs??
        # Last action details of relevance?? -> probably not since fail chance is 0
        - perception (VISION_RANGE x VISION_RANGE x 5) # -1 if value is UNKNOWN -> from the map of the agent
            for each:
            - x
            - y
            - things.type
            - things.details
            - terrain
        - known dispensers (DISPENSER_NUM x 3)
            - x
            - y
            - details
        - attached (TASK_SIZE+2 x TASK_SIZE+2 x 3)
            for each:
            - x
            - y
        - energy (0-MAX_ENERGY) # current energy level
        - step (0-STEP_NUM) # current step number
        - tasks (task_num x (2 + TASK_SIZE x 3)
            - deadline (0-STEP_NUM)
            - reward (0-100)
            - requirements (TASK_SIZE)
                - x
                - y
                - details
        - known walls (WALL_NUM x 2)
            - x
            - y

    ### Actions
    0. - skip
    1. - move[n]
    2. - move[s]
    3. - move[e]
    4. - move[w]
    5. - attach[n]
    6. - attach[s]
    7. - attach[e]
    8. - attach[w]
    9. - detach[n]
    10. - detach[s]
    11. - detach[e]
    12. - detach[w]
    13. - rotate[cw]
    14. - rotate[ccw]
    15. - request[n]
    16. - request[s]
    17. - request[e]
    18. - request[w]
    19 -> 19+task_num. - submit[0-task_num]

        
        # Currently not used actions
        - connect (x, y, agent)
        - disconnect (x1, y1, attachment1, x2, y2, attachment2)
        - clear
"""
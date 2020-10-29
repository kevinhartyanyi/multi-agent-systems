
"""{'lastActionParams': ['w'], 'score': 0, 'lastAction': 'move', 'things': [{'x': 0, 'y': 0, 'details': 'A',
'type': 'entity'}], 'attached': [], 'disabled': False, 'terrain': {'obstacle': [[3, 2], [2, 2], [1, 2], [0, 2], [-1,
2], [-2, 2], [-5, 0], [-3, 2], [1, -4], [0, -4], [-1, -4]]}, 'lastActionResult': 'success', 'tasks': [{'reward': 10,
'requirements': [{'x': 0, 'y': 1, 'details': '', 'type': 'b2'}], 'name': 'task11', 'deadline': 408}, """


"""
Rewards:
    0. - skip:
        -1 always or remove this action

    1. - move[n]
    2. - move[s]
    3. - move[e]
    4. - move[w]:
        +2 if the agent has any blocks attached that are required by any active task
        +1 if the agent has any blocks attached
        -1 if fails
        0 otherwise

    5. - attach[n]
    6. - attach[s]
    7. - attach[e]
    8. - attach[w]
        //Can't stack//
        +20 if successful and there's an active task that requires that block
        +10 if successful
        +5 if fails, but there's a block in 1 distance (L1) from the agent
        +3 if fails, but there's a block in 2 distance (L1) from the agent
        +1 if fails, but there's a block in 3 distance (L1) from the agent
        -1 otherwise

    9. - detach[n]
    10. - detach[s]
    11. - detach[e]
    12. - detach[w]
        //Can't stack//
        +10 if successful
        +5 if fails, but the agent has any blocks attached
        -1 otherwise

    13. - rotate[cw]
    14. - rotate[ccw]
        -1 if agent doesn't have any blocks attached
        0 otherwise

    15. - request[n]
    16. - request[s]
    17. - request[e]
    18. - request[w]
        //Can't stack//
        +10 if successful
        +5 if fails, but there's a dispenser in 1 distance (L1) from the agent
        +3 if fails, but there's a dispenser in 2 distance (L1) from the agent
        +1 if fails, but there's a dispenser in 3 distance (L1) from the agent
        -1 otherwise

    19 -> 19+task_num. - submit[0-task_num]
        //Can't stack//
        +100 + task_point if successful
        +60 if agent has any blocks attached and stands on a goal cell
        +50 if agent has any blocks attached and the closest goal cell from the agent is 1 distance (L1) away
        +40 if agent has any blocks attached and the closest goal cell from the agent is 2 distance (L1) away
        +30 if agent has any blocks attached and the closest goal cell from the agent is 3 distance (L1) away
        +20 if agent doesn't have any blocks attached and stands on a goal cell
        +10 if agent doesn't have any blocks attached and the closest goal cell from the agent is 1 distance (L1) away
        +5 if agent doesn't have any blocks attached and the closest goal cell from the agent is 2 distance (L1) away
        +3 if agent doesn't have any blocks attached and the closest goal cell from the agent is 3 distance (L1) away

//Can't stack//
    If the agent consecutively does the same action t times (where t is a threshold hyperparameter),
    then the agent will receive 0 points for that action in the next 5? rounds. This doesn't apply if the action was
    successful.
"""
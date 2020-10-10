terrain_dict = {
    0: 'empty',
    1: 'goal',
    2: 'obstacle'
}

_things_dict = {
    'empty':    {
                'code': 0
                },
    'entity':   {
                'code': 1,
                'details':
                    {
                    'A': 0,
                    'B': 1
                    }
                },
    'block':    {
                'code': 2,
                'details':
                    {
                    'b0': 0,
                    'b1': 1,
                    'b2': 2
                    }
                },
    'dispenser':{
                'code': 3 # Duplicate 'block'?
                },
    'marker':   {
                'code': 4,
                'details':
                    {
                    'clear': 0,
                    'ci': 1,
                    'cp': 2
                    }
                }
}









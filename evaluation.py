import json

with open('movies/evaluation.json') as json_file:  
    evaluations = json.load(json_file)

    f1 = f2 = f3 = f4 = f5 = f6 = f7 = f8 = f9 = f10 = counter = 0
    
    for evaluation in evaluations:
        if((evaluation['1'] >= 0.0) | 
            (evaluation['2'] >= 0.0) | 
            (evaluation['3'] >= 0.0)  | 
            (evaluation['4'] >= 0.0)  | 
            (evaluation['5'] >= 0.0)  | 
            (evaluation['6'] >= 0.0)  | 
            (evaluation['7'] >= 0.0)  | 
            (evaluation['8'] >= 0.0) | 
            (evaluation['9'] >= 0.0)  | 
            (evaluation['10'] >= 0.0) ):
            print(evaluation)
            f1 = f1 + evaluation['1']
            f2 = f2 + evaluation['2']
            f3 = f3 + evaluation['3']
            f4 = f4 + evaluation['4']
            f5 = f5 + evaluation['5']
            f6 = f6 + evaluation['6']
            f7 = f7 + evaluation['7']
            f8 = f8 + evaluation['8']
            f9 = f9 + evaluation['9']
            f10 = f10 + evaluation['10']
            counter = counter + 1

    print('Counter = ', counter)
    
    print('Average F1 = ', f1/counter)
    print('Average F2 = ', f2/counter)
    print('Average F3 = ', f3/counter)
    print('Average F4 = ', f4/counter)
    print('Average F5 = ', f5/counter)
    print('Average F6 = ', f6/counter)
    print('Average F7 = ', f7/counter)
    print('Average F8 = ', f8/counter)
    print('Average F9 = ', f9/counter)
    print('Average F10 = ', f10/counter)

    average = ((f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10)/10)/counter

    print('Average = ', average)
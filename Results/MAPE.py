import re
import csv

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

file1 = open('random.txt', 'r')
Lines = file1.readlines()
 
horizon10=[0.009798845208230015, 0.010008374606281827, 0.013746961505450591, 0.01144433057436862, 0.01316334815606416, 0.014810359834587805, 0.015379551435126608, 0.017549473647433183, 0.017314210370024685, 0.01685871381289054]
horizon20=[0.019699095104683442, 0.018658373369884407, 0.022224455066162233, 0.014881787252223137, 0.011561816310989376, 0.011240934938803687, 0.012749496613611144, 0.01965686636733204, 0.01719364534159697, 0.009301115506623943, 0.009798845208230015, 0.010008374606281827, 0.013746961505450591, 0.01144433057436862, 0.01316334815606416, 0.014810359834587805, 0.015379551435126608, 0.017549473647433183, 0.017314210370024685, 0.01685871381289054]
horizon30=[0.019446072545896642, 0.023124960151588672, 0.023145691549789623, 0.011728508990211172, 0.010928682841968535, 0.01610044993324806, 0.016503484452868755, 0.0190315451295163, 0.020526663038328136, 0.01794591491670361, 0.019699095104683442, 0.018658373369884407, 0.022224455066162233, 0.014881787252223137, 0.011561816310989376, 0.011240934938803687, 0.012749496613611144, 0.01965686636733204, 0.01719364534159697, 0.009301115506623943, 0.009798845208230015, 0.010008374606281827, 0.013746961505450591, 0.01144433057436862, 0.01316334815606416, 0.014810359834587805, 0.015379551435126608, 0.017549473647433183, 0.017314210370024685, 0.01685871381289054]


with open('random.csv','w') as file:
    for line in Lines:
        horizon=0
        dModel=0
        learningRate=0.0
        heads=0
        encodingLayers=0
        if "horizon" in line:
            parameters = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            parameters = [str(number) for number in parameters]
            print(parameters)
            horizon = parameters[0]
            dModel = parameters[1]
            if len(parameters)==7:
                learningRate =  parameters[2]+"e-0"+parameters[3]
                slidingrate=parameters[4]
                heads = parameters[5]
                encodingLayers = parameters[6]
            else:
                learningRate = parameters[2]
                slidingrate=parameters[3]
            slidingrate = parameters[4]
            heads = parameters[5]
            encodingLayers = parameters[6]
            file.write(horizon+",")
            file.write(dModel+",")
            file.write(learningRate+",")
            file.write(heads+",")
            file.write(encodingLayers+",")

        
        if "Early stop test" in line:
            thing = line.split("=")
            newThing = thing[1].replace('[',"").replace(']','')
            newList = newThing.strip('][\n').split(', ')
            
            newList = [float(item) for item in newList]
            predictedActual=[]
            print(newList)
            if len(newList)==10:
                predictedActual=horizon10
            elif len(newList)==20:
                predictedActual=horizon20
            else:
                predictedActual=horizon30
            mse =  mean_squared_error(newList, predictedActual)
            mae = mean_absolute_error(newList, predictedActual)
            mape = mean_absolute_percentage_error(newList,predictedActual)
            print(mse)
            file.write(str(mse)+",")
            file.write(str(mae)+",")
            file.write(str(mape)+",")
            file.write('\n')


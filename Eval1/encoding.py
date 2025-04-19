import pandas as pd
import numpy as np



def encode():

    df = pd.concat([pd.read_csv("class_pos_covid.csv"), pd.read_csv("class_pos_covid2.csv")])
    df.fillna(np.nan, inplace=True)
    # print(df.head())

    # keys = ['State', 'Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'RemovedTeeth', 'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory', 'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos']
    types = list(df.dtypes)
    encoders = {}
    enconding = {
        'State': "One-Hot",
        'Sex': "Binary", # Binary/Label?
        'GeneralHealth': "Ordinal",
        'PhysicalHealthDays': None,
        'MentalHealthDays': None,
        'LastCheckupTime': "Ordinal", # Ordinal/Label?
        'PhysicalActivities': "Binary",
        'SleepHours': None,
        'RemovedTeeth': "Ordinal", # Ordinal/Label?
        'HadHeartAttack': "Binary",
        'HadAngina': "Binary",
        'HadStroke': "Binary",
        'HadAsthma': "Binary",
        'HadSkinCancer': "Binary",
        'HadCOPD': "Binary",
        'HadDepressiveDisorder': "Binary",
        'HadKidneyDisease': "Binary",
        'HadArthritis': "Binary",
        'HadDiabetes': "Label", # Ordinal/Label?
        'DeafOrHardOfHearing': "Binary",
        'BlindOrVisionDifficulty': "Binary",
        'DifficultyConcentrating': "Binary",
        'DifficultyWalking': "Binary",
        'DifficultyDressingBathing': "Binary",
        'DifficultyErrands': "Binary",
        'SmokerStatus': "Label", # Ordinal/Label?
        'ECigaretteUsage': "Label", # Ordinal/Label?
        'ChestScan': "Binary",
        'RaceEthnicityCategory': "One-Hot",
        'AgeCategory': "Ordinal", # Ordinal/Label?
        'HeightInMeters': None,
        'WeightInKilograms': None,
        'BMI': None,
        'AlcoholDrinkers': "Binary",
        'HIVTesting': "Binary",
        'FluVaxLast12': "Binary",
        'PneumoVaxEver': "Binary",
        'TetanusLast10Tdap': "Label", # Label/One-Hot?
        'HighRiskLastYear': "Binary",
        'CovidPos': "Binary"
    }
    keys = list(enconding.keys())

    for key in range(len(keys)):
        if types[key] == np.float64:
            encoders[keys[key]] = None
            continue

        unique = list(df[keys[key]].unique())
        if np.nan in unique:
            unique.remove(np.nan)
        unique.sort()
        
        print(keys[key], unique)
        
        encoder = {np.nan: -1}
        for j in range(len(unique)):
            encoder[unique[j]] = j
        encoders[keys[key]] = encoder


    for key, value in encoders.items():
        # print(key)
        if type(value) == dict:
            # for i in value:
            #     print("\t", i, value[i])
            df[key] = df[key].map(value)
        # elif value == None:
        #     print("\t", "float")
    
    # df.to_csv("encoded.csv", index=False)
    # print(df)

    for el in range(len(df.columns)):
        for i in range(el+1, len(df.columns)):
            corr = df[df.columns[el]].corr(df[df.columns[i]])
            if abs(corr) > 0.35:
                print(df.columns[el], df.columns[i], corr)

    return


encode()
import pandas as pd

class Utils(object):
    #data file path
    
    
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                          'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    
    target_fields = ['cnt', 'casual', 'registered']
    
    scaled_features = {}
    
    def __init__(self,data_path='Bike-Sharing-Dataset/hour.csv'):
        self.data_path = data_path
        self.getData()
        self.prepareData()
        self.extractTestAndValidation()
    
    def getData(self):
        self.rides = pd.read_csv(self.data_path)
        
    def prepareData(self):
        #Convert categorical
        
        for each in self.dummy_fields:
            dummies = pd.get_dummies(self.rides[each], prefix=each, drop_first=False)
            rides = pd.concat([self.rides, dummies], axis=1)
        
        self.data = rides.drop(self.fields_to_drop, axis=1)
        
        for each in self.dummy_fields:
            dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
            rides = pd.concat([rides, dummies], axis=1)        
        
        self.data = rides.drop(self.fields_to_drop, axis=1)
        
        #scale features
        # Store scalings in a dictionary so we can convert back later
        
        for each in self.quant_features:
            mean, std = self.data[each].mean(), self.data[each].std()
            self.scaled_features[each] = [mean, std]
            self.data.loc[:, each] = (self.data[each] - mean)/std
            
    def extractTestAndValidation(self):
        #splitting data into training, validation, test
        # Save data for approximately the last 21 days 
        self.test_data = self.data[-21*24:]
        
        # Now remove the test data from the data set 
        self.data = self.data[:-21*24]
        
        # Separate the data into features and targets
        features, targets = self.data.drop(self.target_fields, axis=1), self.data[self.target_fields]
        self.test_features, self.test_targets = self.test_data.drop(self.target_fields, axis=1), self.test_data[self.target_fields]
        
        # Hold out the last 60 days or so of the remaining data as a validation set
        self.train_features, self.train_targets = features[:-60*24], targets[:-60*24]
        self.val_features, self.val_targets = features[-60*24:], targets[-60*24:]
        
        
    
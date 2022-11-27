#Home Environment
import json
import re
import numpy as np
import torch


kilohour_to_sec = 1/60/60/1000
hour_to_sec = 1/60/60


class Home_Environment():

    def __init__(self, step_period, data):
        self.data = data
        self.day_index = 0 #Day index for giving rewards
        self.home_charge = 0 #Home charge in AmpereHours
        self.solar_effect = 10 #Current Solar Effect in Watt
        self.load = 1 #Watt Load from house usage
        self.total_sold_price = 0
        self.total_bought_price = 0
        self.max_charge = 25200*0.95 #max charge in Ampere Seconds
        self.exchange = []
        self.charge_at_time = []
        self.sold_in_day = 0
        self.bought_in_day = 0
        self.average_price_day = 0
        self.sold_prices = []
        self.last_sold_from_battery = 0
        self.bought_at_price = 0
        self.sold_at_price = 0
        self.amount_sold = 0
        self.amount_gained = 0
        self.full_charged = 0

        #Temporary charge and discharge rates until we do better calculations
        self.home_discharge_rate_temp = 1 #Temp 5 ampere discharge (in seconds don't need to be adjusted to period)
        self.solar_charge_rate_temp = self.watt_to_ampere(self.solar_effect) #Temp 2.16 Ampere assumes full charging (in seconds don't need to be adjusted to period)
        self.step_period = step_period #1 is step every second 10 would be step every 10 seconds
        

        time = re.findall(r'\d',data[int(self.day_index*24)]['TimeStampHour'])

        if int(time[2]) == 0:
            self.hour = int(time[3])
        else:
            self.hour = int(time[2])*10+int(time[3])

        if int(time[2]) == 0:
            self.time = int(time[0])
        else:
            self.time = int(time[0])*10+int(time[1])

        self.options = {"forced_solar" : 0}

    def get_options(self):
        '''Options available are: forced_solar (1 for forced only single wattage or 0 for solar power simulated) '''
        return self.options

    def set_option(self, name, value):
        '''Options available are: forced_solar (1 for forced only single wattage or 0 for solar power simulated) '''
        self.options[name] = value
        return self.options

    def sell(self,watt_to_sell):
        gained = self.step_period*(watt_to_sell*kilohour_to_sec)*self.data[int(self.time/3600)]['Value'] #DeltaTime * W * KwhToSec * Price
        self.total_sold_price += gained
        self.sold_in_day += gained
        self.sold_prices.append(self.data[int(self.time/3600)]['Value'])# Append price it sold at
        self.amount_sold += watt_to_sell
        self.amount_gained += gained
        #print(f"Gained {gained} from {watt_to_sell}Watt with value {self.data[int(self.time/3600)]['Value']} | Day count: {self.day_index}")

    def buy(self,watt_to_buy):
        cost = self.step_period*(watt_to_buy*kilohour_to_sec)*self.data[int(self.time/3600)]['Value'] #DeltaTime * Kwh * KwhToSec * Price
        self.total_bought_price += cost
        self.bought_in_day += cost

    def watt_to_ampere(self, target):
        target = target*0.0833333
        return target

    def charge(self,target, charge_amount):
        target += self.step_period*(charge_amount*hour_to_sec)
        return target
    

    def run_timeframe(self, choice):

        #if day is between 22-08 solar power is 1watt
        if not self.options['forced_solar']:
            if self.time >= (79200 + 86400*(self.day_index)) or self.time <= (28800 + 86400*(self.day_index)): 
                self.solar_effect = 1
                self.solar_charge_rate_temp = self.watt_to_ampere(self.solar_effect)
            else:
                self.solar_effect = 5
                self.solar_charge_rate_temp = self.watt_to_ampere(self.solar_effect)

        if self.time >= 86400*(self.day_index+1): #If day is over needs to go to next day and save information
            #print(f"Bought: {self.bought_in_day} | Sold: {self.sold_in_day} ")
            self.day_index += 1

            self.charge_at_time.append(self.home_charge/3600)
            self.exchange.append((self.total_sold_price-self.total_bought_price)/100)
            self.sold_in_day = 0
            self.bought_in_day = 0
            #self.run_timeframe(choice) #Made Environment Rerun day
            #print(f"Time: {self.time} | Sold for: {self.total_sold_price/100} | Bought for: {self.total_bought_price/100} | Day count: {self.day_index} ")
            #return #Made day quit

        self.amount_sold = 0
        self.amount_gained = 0

        if choice == 0:#Home -> Away AND Solar -> Away (SELL)
            #If can sell charge sell it
            if self.home_charge - self.home_discharge_rate_temp*self.step_period > 0:
                self.sell(self.home_discharge_rate_temp*12 + self.solar_effect)
                self.home_charge -= self.home_discharge_rate_temp*self.step_period
            else: 
                self.sell(self.solar_effect)
            self.buy(self.load)
            self.sold_at_price = self.data[int(self.time/3600)]['Value']
            self.bought_at_price = 0
            
        elif choice == 1:#Away -> Home AND Solar -> Home (BUY)
            #If can charge since not full
            self.full_charged = 1
            if self.home_charge + self.home_discharge_rate_temp*self.step_period + self.solar_charge_rate_temp*self.step_period < self.max_charge:
                self.home_charge += self.home_discharge_rate_temp*self.step_period + self.solar_charge_rate_temp*self.step_period
                self.buy((self.home_discharge_rate_temp*12))
                self.full_charged = 0
            elif self.home_charge < self.max_charge: #If goes over max needed due to large steps
                home_plus_solar = self.home_charge + self.watt_to_ampere(self.solar_effect)*self.step_period
                if home_plus_solar > self.max_charge:
                    amount_to_sell = self.watt_to_ampere(self.solar_effect)*self.step_period - (self.max_charge - self.home_charge)
                    self.sell((amount_to_sell*12)/self.step_period)
                    self.home_charge = self.max_charge
                else:
                    self.buy((self.max_charge-self.home_charge)/3600)
                self.home_charge = self.max_charge     
            self.buy(self.load)
            self.bought_at_price = self.data[int(self.time/3600)]['Value']
            self.sold_at_price = 0
            

        if not self.options['forced_solar']:
            #if day is between 22-08 in one hour change power
            if self.time+self.step_period >= (79200 + 86400*(self.day_index)) or self.time+self.step_period <= (28800 + 86400*(self.day_index)): 
                self.solar_effect = 1
                self.solar_charge_rate_temp = self.watt_to_ampere(self.solar_effect)
            else:
                self.solar_effect = 5
                self.solar_charge_rate_temp = self.watt_to_ampere(self.solar_effect)
        
        self.time = self.step_period + self.time
        self.last_sold_from_battery += self.step_period
        
    def day_init(self):
        state = []
        day_arr = []
        for i in range(24):
            day_arr.append(self.data[int(i+(self.time/3600))]['Value'])
        for i in range(3):
            state.append(self.data[int(i+(self.time/3600))]['Value'])
        if self.sold_prices != []:
            self.avg_sold_price = np.mean(self.sold_prices[-24*7:]) #Gets average sold price at last 24 sold prices
        self.avg_price_day = np.mean(day_arr, dtype=np.float32)
        state.append(100/(self.max_charge/(self.home_charge+1)))
        #state.append(self.data[int(self.time/3600)]['Value'])
        #state.append(self.home_charge)
        #state.append(self.max_charge)
        state.append(self.solar_effect)

        #state.append(self.data[int(self.time/3600)]['Value'])
        #state.append(self.load)
        #state.append(self.average_price_day)
        #state.append(self.solar_charge_rate_temp)
        #state.append(0.0)
        #state.append(self.max_charge)
        
        return state

    def step(self, choice):
        done = 0
        dayarr = re.findall(r'\d',self.data[self.day_index]['TimeStampDay'])
        day = int(dayarr[6])*10 + int(dayarr[7])
        initial_date = day
        if day == initial_date:
            self.run_timeframe(choice)
        if self.time >= 86400*(self.day_index+1): #If going to next day
            done = 1
        
        if self.day_index > (len(self.data)/24) - 2: #Returns if the max day is reached which as of current is day length -1 (0 is included)
            return 0, 0, 0
        
        state = []
        day_arr = []
        #if (self.day_index > 674):
        #    print(self.day_index+1)
        #    print(self.data[int(23+(self.time/3600))]['TimeStamp'])
        for i in range(24):
            day_arr.append(self.data[int(i+(self.time/3600))]['Value'])
        for i in range(3):
            state.append(self.data[int(i+(self.time/3600))]['Value'])
        #if len(self.sold_prices) > 0:
        #    self.avg_sold_price = np.mean(self.sold_prices[-24*7:]) #Gets average sold price at last 24 sold prices
        #else:
        #    self.avg_sold_price = self.data[int(self.time/3600)]['Value']

        self.avg_price_day = np.mean(day_arr, dtype=np.float32) #Gets average price of next 24 
        state.append(self.home_charge/3600)
        state.append(self.solar_effect)#Remove if model2

        #If Sold
        if self.bought_at_price == 0:
            #reward = ((self.sold_at_price/(self.avg_price_day+0.01))-1) #Sold at price compared to average price
            reward = (day_arr[0] - day_arr[1])*2 + (day_arr[1] - day_arr[2])*1
        else: #If Bought
            #reward = ((self.avg_price_day/self.bought_at_price)-1) #Ignored price comparde to average price
            reward = (-day_arr[0] + day_arr[1])*2 + (-day_arr[1] + day_arr[2])*1
        reward = reward*(self.step_period/3600)
        reward += (self.sold_in_day - self.bought_in_day)*0.4*(self.step_period/3600)*15
        
        

        #reward = self.total_sold_price - self.total_bought_price

        #reward += (self.full_charged)*-reward

        #optimal_solar_sell = (self.step_period*(self.solar_effect*kilohour_to_sec)*self.data[int(self.time/3600)]['Value'])
        #reward = (self.amount_gained - optimal_solar_sell)*10
        #reward += self.home_charge/(3600*3)
        #reward += self.full_charged*-50
        #reward = self.sold_in_day - self.bought_in_day
        #print(reward)

        
        return state, reward, done

if __name__ == "__main__":
    #file = open('2008till2022-10-15.json')
    file = open('2021till2022nov.json')
    data = json.load(file)

    print(type(data))
    print(data[0])
    print(data[0]['TimeStampHour'])
    print(data[0]['TimeStampDay'])
    print(data[0]['Value'])
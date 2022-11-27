import unittest
import homeenv
import numpy as np
import time

Home_Environment = homeenv.Home_Environment

#{'TimeStamp': '2008-01-01T00:00:00', 
# 'TimeStampDay': '2008-01-01', 
# 'TimeStampHour': '00:00', 
# 'Value': 40.00, 
# 'PriceArea': 'SN4', 
# 'Unit': 'Ã¶re/kWh'}



file = open('2008till2022-10-15.json')
#data = json.load(file)
def generate_data(days):
    '''Generate amount of days'''
    start_time = time.time()
    print("Generating Data")
    data = {}
    for j in range(days):
        for i in range(24):
            time_stamp_string = "2008-01-" +str(int(j/10)) +str(j%10)+"T"+str(int(i/10))+str(i%10)+":00:00"
            data[i+j*24] = {'TimeStamp': time_stamp_string, 
                'TimeStampDay': "2008-01-"+str(int(j/10)) +str(j%10), 
                'TimeStampHour': str(int(i/10))+str(i%10)+":00",
                'Value': 40, 
                'PriceArea': 'SN4', 
                'Unit': 'Ã¶re/kWh'}
    print("Finished Generating Data finished after: " + (str(np.round((time.time() - start_time),3))) + " seconds")
    return data



class TestHomeEnv(unittest.TestCase):
    def test_solar_calc(self):
        '''Tests solar effect selling calculation'''
        start_time = time.time()
        print("Testing Solar Effect Calculations")
        for i in range(4):
            if i == 0:
                step = 60*15 #Test 15 min tick
            elif i == 1:
                step = 60*5 #Test 5 min tick
            elif i == 2:
                step = 60 #Test minute tick
            elif i == 3:
                step = 10 #Test 10 second tick
            env = Home_Environment(step,data) #one hour per tick test
            env.solar_effect = 175 # 175 watt
            env.home_discharge_rate_temp = 0
            env.set_option("forced_solar", 1)
            value = (175*0.040/1/60/60/1000)
            value_times_step = step*value*1000
            T1 = 1
            for i in range(T1):
                env.step(0) #step sell and should be 175 * 40 * 0.001 = 7 öre aka 0.07kr
            self.assertEqual(np.round(env.total_sold_price, 6), np.round(value_times_step*T1, 6))
            T2 = 23
            for i in range(T2):
                env.step(0)
            self.assertEqual(np.round(env.total_sold_price,6), np.round((value_times_step*(T1+T2)),6))
            T3 = 1
            for i in range(T3):
                env.step(0)
            self.assertEqual(np.round(env.total_sold_price,6), np.round((value_times_step*(T1+T2+T3)),6))
            env.solar_effect = 50 # 50 watt
            value_times_step2 = (50*40*0.001/60/60)*step
            
            T4 = 1000
            for i in range(T4):
                env.step(0)
            self.assertEqual(np.round(env.total_sold_price,6), np.round((value_times_step2*T4+value_times_step*(T1+T2+T3)),6))

        print("Finished Testing Solar Calculation after: " + (str(np.round((time.time() - start_time),3))) + " seconds")

    def test_charging(self):
        '''Tests charging function'''
        start_time = time.time()
        print("Testing Charging Calculations")
        for i in range(4):
            if i == 0:
                step = 1 #Test 1 sec tick
            elif i == 1:
                step = 2 #Test 2 sec tick
            elif i == 2:
                step = 3 #Test 3 sec tick
            elif i == 3:
                step = 4 #Test 10 second tick
            env = Home_Environment(step,data) #one hour per tick
            env.solar_effect = 120 # 175 watt
            env.solar_charge_rate_temp = 120/12
            env.home_charge = 0
            env.set_option("forced_solar", 1)
            env.home_discharge_rate_temp = 5
            value = ((120/12)+5)
            value_times_step = step*value
            T1 = 1
            for i in range(T1):
                env.step(1) #Charges
            self.assertEqual(np.round(env.home_charge, 6), np.round(value_times_step*T1, 6))
            T2 = 23
            for i in range(T2):
                env.step(1)
            self.assertEqual((env.home_charge), np.round(value_times_step*(T1+T2), 6))
            T3 = 100
            for i in range(T3):
                env.step(1)
            self.assertEqual((env.home_charge), np.round(value_times_step*(T1+T2+T3), 6))
            env.solar_effect = 50 # 50 watt
            T4 = 2000
            env.solar_effect = 10 # 175 watt
            env.solar_charge_rate_temp = 10/12
            env.home_discharge_rate_temp = 1
            value2 = ((10/12)+1)
            value_times_step2 = step*value2
            for i in range(T4):
                env.step(1)
            self.assertEqual(np.round(env.home_charge,6), np.round((value_times_step2*T4+value_times_step*(T1+T2+T3)),6))
        print("Finished Testing Charging Calculation after: " + (str(np.round((time.time() - start_time),3))) + " seconds")

    def test_charging_cap(self):
        '''Tests charging capping function'''
        start_time = time.time()
        print("Testing Charging Capped and Uncapped Calculations")
        for i in range(4):
            if i == 0:
                step = 60*15 #Test 15 min tick
            elif i == 1:
                step = 60*5 #Test 5 min tick
            elif i == 2:
                step = 1 #Test minute tick
            elif i == 3:
                step = 10 #Test 10 second tick
            env = Home_Environment(step,data) #sets tick and resets enviroment
            env.solar_effect = 120 # 175 watt
            env.solar_charge_rate_temp = 120/12
            env.home_charge = 0
            env.set_option("forced_solar", 1)
            env.home_discharge_rate_temp = 5
            value = ((120/12)+5)
            value_times_step = step*value
            T1 = 1
            for i in range(T1):
                env.step(1) #Charges
            self.assertEqual(np.round(env.home_charge, 6), np.round(value_times_step*T1, 6))
            T2 = 3005
            for i in range(int(T2/step)):
                env.step(1)
            self.assertEqual(env.home_charge, env.max_charge)
            T3 = int(20/step)
            value2 = (5)
            value_times_step2 = step*value2
            for i in range(T3):
                env.step(0)
            self.assertEqual(env.home_charge, env.max_charge-(np.round(value_times_step2*T3, 6)))
        print("Finished Testing Charging Capping Calculation after: " + (str(np.round((time.time() - start_time),3))) + " seconds")

def perform_tests():
    home_test = TestHomeEnv()
    home_test.test_solar_calc()
    home_test.test_charging()
    home_test.test_charging_cap()

if __name__ == "__main__":
    data = generate_data(2000)
    perform_tests()
    
#!/usr/bin/env python3
# vim: set ts=4 foldmethod=marker:

############################# README  ##########################################
#{{{1 Documentation
#
# (i) Time Unit -> 1 hour
#      Hence, time will be stored as t = 1.0, 1.2, 4.5858, ....
#      Meaning t = 1h, t = 1h12min, ...
#
# (ii) All simulation will be based on a nonhomogeneous Poisson Process
#
# (iii) Value: N(t) / t.
#      Here, N(t) is the counting measure which counts how many events
#      happened up to time t. 
#
#      N(t) / t can be interpreted at the limit as
#             lim N(t) / t  ~ mean value of buyers per hour.
#      
# (iv) The main process will be based on the superposition of two Poisson
#      Processes: N1, N2.
#
#        * N1 stands for the interarrival of clients buying 'essential goods'.
#          These items can be considered as food, gas, ...
#
#        * N2 stands for the interarrival of clients buying 'non essential goods'.
#          These items can be considered as eletronic devices, toys, clothes, ...
#
# (v) Intensity rates for N1, N2 in the time set [0, 24h)
#      Following the website
#               https://www.oberlo.com/blog/online-shopping-statistics
#      I will make the two assumptions, which can be changed later, that
#
#           N1(t) / 24h ~ 10 * clientsPopSize * 0.9  / 30 
#           N2(t) / 24h ~      clientsPopSize * 0.62 / 30
#      
#      The values for the rates are based on the following insight: 
#      90% of the client's population buy an essential good at least 10 times per
#      month; meanwhile around 62% of the client's population buy a nonessential
#      good in a month
#
# (vi) Intensity rate, lambda_t(t), in the window [0, 24h)
#      By item (v) I will set that
#
#            int_0^{24} lambda_t(t) dt = (10 * clientsPopSize * 0.9  / 30 + 
#                                              clientsPopSize * 0.62 / 30 )^{-1}
#1}}}

################# MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import time 

class simulationAssumptions:
	"""
		A class that provides all assumptions used at the simulation

		Atributes
			* clientsPopSize

			* storesPopSize

			* ball_radius_normLoo
				A variable whose value R indicates that the whole population of
				clients are inside the closed ball

					B[0, R] = {x in R^2; |x| = max{x_1, x_2} <= R}

		Methods:

			* pdf_rate_func()
			* wageDist()
			* plot_wage()
			* client_locDist()
			* max_dist_from_home()
			* creditCard_limit()
	"""

	#{{{1 Attributes
	def __init__(self,
				 clientsPopSize,
				 storesPopSize,
				 ball_radius_normLoo):

		self.clientsPopSize      = clientsPopSize

		self.storesPopSize              = storesPopSize
		self.amount_essential_stores    = int(.7 * storesPopSize)
		self.amount_nonessential_stores = int(.3 * storesPopSize)

		# mean frequency someone buys an essential product per month
		self.frequencyN1 = 10 

		# mean frequency someone buys a nonessential product per month
		self.frequencyN2 = 1

		self.ball_radius_normLoo = ball_radius_normLoo

		# rates for the thining poisson processes
		## essential goods rate
		self.essential_rate    = self.clientsPopSize * 10 * 0.9 / 30 

		## nonessential goods rate
		self.nonessential_rate = self.clientsPopSize *     0.62 / 30 

		## fraud rate
		self.fraud_rate        = self.clientsPopSize *     0.05 / 30 
	#1}}}

	#{{{1 Methods
	def pdf_rate_func(self):
		"""
			Function that defines a pdf for the intensity rates on [0, 24). 
			It's value is used on the lambda_t(t) function on the
			nonhomogeneous Poisson Process

			The pdf must be a step function 
			 
				sum_{i=0}^{23} r_i 1_{[i, i+q)}

			which, in use with pdf_rate_func, returns an np.array
			of its image (cardinality = 24)
		"""
		#{{{2 function scope
		#First calculate a nice probability density function for lambda_t
		x = np.arange(1, 25)
		def weights (x):
			if x <= 15:
				return 1 / (1 + np.exp(-(x-8)))
			else:
				return 1 / (1 + np.exp((x-22)))

		pdf_rate_Im = np.array( list(map(weights, x)) )
		pdf_rate_Im /= pdf_rate_Im.sum()

		return pdf_rate_Im
		#2}}}


	def integral_0_24_of_lambda_t(self):
		"""
			Assumption about the value of the  integral of the intensity function
					int_0^{24} lambda_t(t) dt
		"""
		#{{{2 function scope 

		# (N(t) / t)^{-1} rates estimative on the window [0, 24h)
		K  = self.essential_rate + self.nonessential_rate + self.fraud_rate 

		return K
		#2}}}


	def wageDist(self):
		"""
			Return a generator where at each next() comand it returns
			a value sampled from a distribution D.
			The distribution D is associated with a random varible that measures
			the client's wage per month. 

			The distribution used here is a gamma distribution 

				W ~ gamma(alpha, betta)

			so that

				E[W] = 69.4 * 10**3 / 12  and Var[W] = 4_000**2.

			That is
				a = E[W]**2 / Var[W]    and scale = Var[W] / E[W]
		"""
		#{{{2 function scope
		E   = 69.4 * 10**3 / 12
		Var = 3_800**2

		a     = E**2 / Var
		scale = Var / E

		seed = int(time.time() * hash('wageDist')) & (2**32 -1)

		dist              = stats.gamma(a, scale=scale)
		dist.random_state = seed

		while True:
			yield dist.rvs()
		#2}}}


	def plot_wage(self):
		#{{{2 function scope
		dist = self.wageDist()

		x0 = dist.ppf(0.01)
		xf = dist.ppf(0.99)

		x = np.linspace(x0, xf, 1000)

		plt.plot(x, dist.pdf(x))
		plt.show()
		#2}}}


	def client_locDist(self):
		"""
			This method  returns a generator object. At each next() command 
			it produces a  random point, representing a localization,
			inside B[0, ball_radius_normLoo]

			the value after the yield is a tuple (x,y), x,y in R
		"""
		#{{{2 funcion scope
		R    = self.ball_radius_normLoo

		dist = stats.uniform(loc=-R/2, scale=R)
		dist.random_state = int(time.time()*hash('client_locDist')) & (2**32 -1)

		while True:
			yield (dist.rvs(), dist.rvs())
		#2}}}


	def max_dist_from_home(self):
		"""
			This function returns a generator and whenever the generator is 
			invocated, by next(), it returns a random variable that has a
			probability distribution P.

			Such distribution will be in charge of randomilly associate to each
			client a radius R. The value R will serve us as an indicator of 
			places that the client is more likely to be.
		"""
		#{{{2 function scope
		R    = self.ball_radius_normLoo
		dist = stats.norm(loc=R/2, scale = R/4)

		dist.random_state = int(time.time()*hash('max_dist_from_home')) & (2**32 - 1) 

		while True:
			yield min([ np.abs(dist.rvs()), R ])
		#2}}}


	def creditCard_limit(self, estimated_wage_per_month):
		"""
			Given an estimated wage per month this method returns
			a credit card limit. The limit will be choosen with
			the assistance of a random variable.
		"""
		#{{{2 function scope
		dist = stats.norm(loc = 1, scale = 0.5)

		#dist.random_state = int(time.time() * hash('creditCard_limit')) & (2**32 - 1) 
		return max([dist.rvs(), 0.15]) * estimated_wage_per_month
		#2}}}


	def distanceGenerator(self):
		"""
			A function that returns a generator whose objective is, at each
			next() command, to return a random tuple (a,b) representing
			a direction to walk
		"""
		#{{{2 function scope
		#seed = int(time.time() * hash('distanceGenerator')) & (2**32 - 1)

		distribution              = stats.norm()
		#distribution.random_state = seed

		while True:
			yield distribution.rvs(size=2)
		#2}}}


	def priceGenerator(self):
		"""
			A function that returns a generator whose objective is, at each
			next() command, to return a random value (a,b) representing
			a possible price
		"""
		#{{{2 function scope
		#seed = int(time.time() * hash('priceGenerator')) & (2**32 - 1)

		distribution              = stats.norm()
		#distribution.random_state = seed

		while True:
			yield distribution.rvs()
		#2}}}

	def  throw_coin(self):
		"""
			The function returns a generator.
			Each next() is equivalent to throwing a coin and returning
			wheter the transaction is:
				* a fraud;
				* an essential buying;
				* a nonessential buying.
		"""
		#{{{2 function scope
		K = self.essential_rate + self.nonessential_rate + self.fraud_rate
		p_fraud               = self.fraud_rate        / K
		p_essential_buying    = self.essential_rate    / K
		p_nonessential_buying = self.nonessential_rate / K

		#seed = int(time.time() * hash('throw_coin')) & (2**32 -1)

		coin              = stats.uniform()
		#coin.random_state = seed

		while True:
			throwCoin = coin.rvs()

			if throwCoin < p_fraud:
				yield 'fraud'
			elif p_fraud <= throwCoin < p_fraud + p_essential_buying:
				yield 'essential'
			else:
				yield 'nonessential'
		#2}}}

	def frauds_in_a_row_generator(self):
		"""
			Returns a generator
			At each next() call a sample from a probability distribution P is 
			returned.

			The distribution P measures the probability 
			of having n frauds in a row
		"""
		#{{{2 function scope
		#seed = int(time.time() * hash("frauds_in_a_row_dist")) & (2**32 -1)

		distribution = stats.geom(0.1)
		#distribution.random_state = seed

		while True:
			yield distribution.rvs()
		#2}}}
	#1}}}





class nonhomogeneous_PoissonProcess:
	"""
		Class responsible to generate all interarrival times from 
		a nonhomogeneous Poisson Process

		Atributes:
			* __assumption_obj:
				an object from the assumption class

			* sup_lambda_t:
				sup of the intensity funcion on the time range [0, 24)
			               

		Methods:
			* lambda_t:
				intensity function

			* plot_lambda_t

			* simulate1day: 
				returns a generator where at each next return the arrival
				time of the nth event

			* sup_lambda_t:
				setters and getters
	"""

	#{{{1 Attributes
	def __init__(self, assumption_obj):

		self.__assumption_obj   = assumption_obj

		self.sup_lambda_t       = None
	#1}}}

	#{{{1 Methods
	def lambda_t(self, t):
		"""
			lambda_t: [0, 24h)  -> R
						 x     |-> sum_{i=0}^{23h} r_i * I_i(x),

			where
				  * I_i(x) = 1 if x in [i, i + 1) and 0 otherwise,
			      * i in {0, 2, 3, ..., 23}
				  * r_i in R all positive real numbers

			This function is the intensity function
		"""
		#{{{2 function scope 

		K           = self.__assumption_obj.integral_0_24_of_lambda_t()
		lambda_t_Im = K * self.__assumption_obj.pdf_rate_func()

		return lambda_t_Im[int(t)]
		#2}}}


	def plot_lambda_t(self):
		"""
			Plot the intensity function
		"""
		#{{{2 function scope 

		K           = self.__assumption_obj.integral_0_24_of_lambda_t()
		lambda_t_Im = K * self.__assumption_obj.pdf_rate_func()

		plt.bar(np.arange(24), lambda_t_Im)
		plt.show()
		#2}}}


	def simulate1day(self):
		"""
			Generator function that at each next() will return an interarrival
			of the respectively nonhomogeneous Poisson Process
		"""
		#{{{2 function scope 
		t_n = 0 # interarrival time after  the n-1th event
		T_n = 0 # T_n = t_1 + t_2 + ... + t_n

		u              = stats.uniform()
		u.random_state = int(time.time()*hash('PoissonProcess')) & (2**32 - 1) 
		while True:
			t_n = - np.log(u.rvs()) / self.sup_lambda_t
			T_n += t_n

			if T_n < 24:
				if u.rvs() <= self.lambda_t(T_n) / self.sup_lambda_t:
					yield T_n

			else:
				return -1 # throws an exception
		#2}}}

	
	@property
	def sup_lambda_t(self):
		"""
			Get the sup value of lambda_t = sup {lambda_t(t); t in [0, 24h)}
		"""
		#{{{2 function scope 
		return self.__sup_lambda_t
		#2}}}


	@sup_lambda_t.setter
	def sup_lambda_t(self, tmp):
		"""
			Set the sup value of lambda_t = sup {lambda_t(t); t in [0, 24h)}
		"""
		#{{{2 function scope 
		K           = self.__assumption_obj.integral_0_24_of_lambda_t()
		lambda_t_Im = K * self.__assumption_obj.pdf_rate_func()

		self.__sup_lambda_t = lambda_t_Im.max()
		#2}}}
	#1}}}





class clients:
	"""
		class responsible to take care of all info regarding the clients
		----------------------------------------------------------------

		Atributes (all Private):
			* __clientsPop  
				List containing all clients profile. It will look like

							clientsPop = [c1, c2, ...]

				where c1, c2, and so on are dictionaries representing the
				ith client. The dictionaries for each client will have the
				following keys

					c1.keys = ['clientID',
							   'estimated_wage_per_month',
					           'localization',
							   'max_dist_from_home',
							   'creditCard_limit',
							   'creditCard_limitUsed',
							   'shopping_list']

				where shopping_list will be a list of dictionaries
					
					c1['shopping_list'] = [b1, b2, ...]

				with
					b1.keys = [ 'buyID',
							    'time',
							    'money_spend',
							    'shop_accepted',
							    'was_a_fraud',
							    'store_bought_from',
							    'type_product',
							    'place_where_cc_was_used']

			* __assumption_obj:
				an object from the assumption class


		Methods

			* __clientsPop:
				initiallize the values of __clientsPop for the simulation

	"""

	#{{{1 Attributes
	def __init__(self, assumption_obj, initialize=False):

		self.__assumption_obj = assumption_obj
		self.__clientsPop     = self.__clientsPop(initialize)
	#1}}}


	#{{{1 Methods
	def __clientsPop(self, initialize):
		"""
			Initiallize the list self.__clientsPop
		"""
		#{{{2 function scope

		# allocating the assumptions
		clientsPopSize        = self.__assumption_obj.clientsPopSize
		R                     = self.__assumption_obj.ball_radius_normLoo
		wageDist              = self.__assumption_obj.wageDist()
		client_locDist        = self.__assumption_obj.client_locDist()
		max_dist_from_home    = self.__assumption_obj.max_dist_from_home()
		creditCard_limit_func = self.__assumption_obj.creditCard_limit


		result = []
		if initialize == False:
			return result

		for clientID in range(clientsPopSize):
			client = {'clientID':                 clientID,
					  'estimated_wage_per_month': next(wageDist),
					  'localization':             next(client_locDist),
					  'max_dist_from_home':       next(max_dist_from_home),
					  'creditCard_limit':         0.0,
					  'creditCard_limitUsed':     0.0,
					  'shopping_list':            []
			}

			wage_per_month             = client['estimated_wage_per_month']
			client['creditCard_limit'] = creditCard_limit_func(wage_per_month)

			result.append(client)

		return result
		#2}}}


	def choose(self):
		""" 
			This function returns a genarator

			After a next() command it will choose randomilly a client
			from __clientsPop and return an client index 

			The function works just like np.random.choose but here
			the probability to choose one of the clients is based on
			how much credit he/she still has

			I suppose that people with higher credit limit are more likely
			to buy something
		"""
		#{{{2 function scope
		clientsPopSize = self.__assumption_obj.clientsPopSize


		seed           = int(time.time() * hash('choose')) & (2**32 -1)
		u              = stats.uniform()
		u.random_state = seed

		while True:
			creditCard_restOfLimit = np.zeros(clientsPopSize)
			for client in self.__clientsPop:
				creditCard_restOfLimit[ client['clientID'] ] = \
					client['creditCard_limit'] - client['creditCard_limitUsed']


			if creditCard_restOfLimit.sum() > 0:
				p  = creditCard_restOfLimit.cumsum()
				p /= creditCard_restOfLimit.sum()

			else:
				return -1 #raise a flag of warning
			
			coin = u.rvs()
			for clientID in range(clientsPopSize):
				if coin < p[clientID]:
					break
				clientID += 1

			yield clientID
		#2}}}


	def buy(self,
	        store_obj,
			clientID,
			distanceGenerator = None,
			priceGenerator    = None,
			product_to_buy_is_essential=True):
		"""
			Returns a tuple 

				(store where client is buying, money spent, type of product)

			where 'type of product' is essential or nonessential
		"""
		#{{{2 function scope

		# assumption values 
		client_i = self.__clientsPop[clientID]

		home_client_i            = np.array( client_i['localization'] )
		estimated_wage_per_month = client_i['estimated_wage_per_month']
		creditCard_limit         = client_i['creditCard_limit'] 
		creditCard_limitUsed     = client_i['creditCard_limitUsed'] 
		Rmaximum                 = client_i['max_dist_from_home']


		# distanceGenerator is a standard normal in R^2
		distance = Rmaximum / 2 * next(distanceGenerator) + 0

		place_client_is_now = home_client_i + distance


		# From his/her position, the client will choose the merchant to buy
		# based on the distance from him/her and the price itself.
		# Now, the problem is merely minimizing a loss function
		type_product = None
		frequency    = 0
		priceMean    = 0
		sigma        = 0
		storesList   = None

		if product_to_buy_is_essential == True:
			frequency    = self.__assumption_obj.frequencyN1
			#priceMean    = estimated_wage_per_month / (3.5 * frequency)
			priceMean    = \
				(creditCard_limit - creditCard_limitUsed) * 0.03
			sigma        = 40
			size_stores  = self.__assumption_obj.amount_essential_stores
			storesList   = store_obj.essential_stores
			type_product = 'essential'
		else:
			frequency   = self.__assumption_obj.frequencyN2
			#priceMean   = (creditCard_limit - creditCard_limitUsed) / (4*frequency)
			priceMean   = (creditCard_limit - creditCard_limitUsed) * 0.5
			sigma       = priceMean / 2
			size_stores = self.__assumption_obj.amount_nonessential_stores
			storesList  = store_obj.nonessential_stores
			type_product = 'nonessential'



		pricesRandom = np.ones(size_stores)
		prices = \
			np.array(
				list( 
					map(lambda x:  x * next(priceGenerator), pricesRandom)
				)
			)

		prices = np.abs( priceMean + sigma * pricesRandom )

		loss_function = np.zeros(size_stores)
		i             = 0

		for store in storesList:
			loc_store = np.array(store['localization'])
			distance  = np.abs(np.max(loc_store - place_client_is_now))

			loss_function[i] = prices[i] * np.log(1 + distance)
			i += 1

		return (np.argmin(loss_function),
				prices[np.argmin(loss_function)],
				type_product)
			
		#2}}}

	def confirm_transaction(self,
							clientID,
							moneySpent,
							typeProduct,
							storeID,
							was_a_fraud,
							place_where_cc_was_used,
							time):
		"""
			Confirm the transactions on the database
		"""
		#{{{2 function scope
	
		client        = self.get_client_i(clientID)
		shopping_list = client['shopping_list']

		buyID         = len(shopping_list) + 1
		shop_accepted = None

		credit_left = client['creditCard_limit']-client['creditCard_limitUsed']

		if moneySpent > credit_left:
			money         = 0.0
			shop_accepted = False
		else:
			money         = moneySpent
			shop_accepted = True
			client['creditCard_limitUsed'] += moneySpent


		shopping_list.append( {'buyID':                  buyID,
							   'time' :                  time,
							   'moneySpent':             money,
							   'shop_accepted':          shop_accepted,
							   'was_a_fraud':            was_a_fraud,
							   'store_bought_from':      storeID,
							   'type_product':           typeProduct,
							   'place_where_cc_was_used':place_where_cc_was_used
		})
		#2}}}


	def plot_loc(self):
		"""
			Plot every client's localization
		"""
		#{{{2 function scope
		loc = np.zeros((self.__assumption_obj.clientsPopSize, 2))

		i = 0
		for client in self.__clientsPop:
			loc[i] = client['localization']
			i += 1

		plt.scatter(loc[:,0], loc[:,1], marker="x", color="red")
		plt.show()
		#2}}}


	def print_client_i(self, client_i):
		#{{{2 function scope
		client = self.__clientsPop[client_i]

		print(f"\nClient {client_i}\n")
		print(f"wage/month {client['estimated_wage_per_month']}" )
		print(f"Limit total {client['creditCard_limit']}")
		print(f"Limit used {client['creditCard_limitUsed']}\n")
		#2}}}

	def get_client_i(self, clientID, keyProperty=None):
		"""
			get the 'keyProperty' key-value from __clientsPop relative
			to client clientID. If keyProperty == None then the dictionary
			relative to clientID is returned
		"""
		#{{{2
		if keyProperty is None:
			return self.__clientsPop[clientID]
		else:
			return self.__clientsPop[clientID]['keyProperty']
		#2}}}
	#1}}}





class stores:
	"""
		Class representing the two kinds of stores I deal with:
			(i) store that sells products that are part of daily consumption
			    of any person: supermarkets, gas station,...

				I will call these stores as "essential_stores"

			(ii) The complementar of above. Stores that sells, as instance,
			    televions, cell phones, cars, ...

				These stores will be called as "nonessential_stores"

		Assumption: all store has a fixed location on the space

		Atributes
			* essential_stores
				A list [s1, s2, ...] with len() == __amount_essential_stores
				Each s1, s2 are dictionaries
				s1.keys = ['storeID',
						   'localization',
						   'type']

			* nonessential_stores
				A list [s1, s2, ...] with len() == __amount_nonessential_stores
				Each s1, s2 are dictionaries
				s1.keys = ['storeID',
						   'localization',
						   'type']
	"""

	#{{{1 Attributes
	def __init__(self, assumption_obj):
		self.__assumption_obj = assumption_obj

		self.__amount_essential_stores    = assumption_obj.amount_essential_stores
		self.__amount_nonessential_stores = assumption_obj.amount_nonessential_stores

		self.essential_stores    = self.set_essential_stores()
		self.nonessential_stores = self.set_nonessential_stores()
	#1}}}

	
	#{{{1 Methods
	def set_essential_stores(self):
		#{{{2 function scope
		R = self.__assumption_obj.ball_radius_normLoo

		seed = int(time.time() * hash('essential_stores')) & (2**32 -1)

		dist              = stats.uniform(loc=-R/2, scale=R)
		dist.random_state = seed

		result = []
		for i in range(self.__amount_essential_stores):
			store = {'storeID':      i,
			         'localization': (dist.rvs(), dist.rvs()),
					 'type'        : 'essential'
			}
			result.append(store)

		return result
		#2}}}


	def set_nonessential_stores(self):
		#{{{2 function scope
		R = self.__assumption_obj.ball_radius_normLoo

		seed = int(time.time() * hash('nonessential_stores')) & (2**32 -1)

		dist              = stats.uniform(loc=-R/2, scale=R)
		dist.random_state = seed

		result = []
		for i in range(self.__amount_nonessential_stores):
			store = {'storeID':      i,
			         'localization': (dist.rvs(), dist.rvs()),
					 'type'        : 'nonessential'
			}
			result.append(store)

		return result
		#2}}}
		
	
	def get_store_i(self, storeID, storeType='essential', keyProperty=None):
		"""
			get the 'keyProperty' key-value from __essential_stores or
			__nonessential_stores referent to storeID. 
		"""
		#{{{2
		if storeType == 'essential':
			return self.essential_stores[storeID][keyProperty]
		else:
			return self.nonessential_stores[storeID][keyProperty]
		#2}}}
	#1}}}






class fraudSimulation:

	#{{{1  Attributes
	def __init__(self,
				 amount_of_days,
				 clientsPopSize,
				 storesPopSize,
				 ball_radius_R):

		self.amount_of_days = amount_of_days

		self.assumptions    = simulationAssumptions(clientsPopSize,
											        storesPopSize,
											        ball_radius_R) 

		self.clientsSim     = clients(self.assumptions, initialize = True)

		self.storesSim      = stores(self.assumptions)

		self.poissonProcess = nonhomogeneous_PoissonProcess(self.assumptions)

	#1}}}

	#{{{1 Methods
	def runSim(self):
		#{{{2 function scope
		amount_of_days = self.amount_of_days
		assumptions    = self.assumptions
		clientsSim     = self.clientsSim
		storesSim      = self.storesSim
		poissonProcess = self.poissonProcess

		print('\n')
		print("Simulating....")
		for day in range(amount_of_days):
			# initialize generators
			choose_client     = clientsSim.choose()
			distanceGenerator = assumptions.distanceGenerator()
			priceGenerator    = assumptions.priceGenerator()
			simGenerator      = poissonProcess.simulate1day()
			coinGenerator     = assumptions.throw_coin()
			frauds_in_a_row   = assumptions.frauds_in_a_row_generator()

			if day % 30 == 0:
				self.payDebtEndMonth()

			while True:
				try:
					T_n       = next(simGenerator)
					coin      = next(coinGenerator)
					client_i  = next(choose_client)

					if coin == 'fraud':
						sequenceTimes = next(frauds_in_a_row)

						#seedFraud = int(time.time()) & (2**32 -1)

						typeStore              = stats.bernoulli(0.5)
						#typeStore.random_state = seedFraud

						for i in range(sequenceTimes):
							#print("\t\tFrauds:", sequenceTimes)
							N              = 1
							storeList      = []
							typeProduct = None

							if typeStore.rvs() == 0:
								N = assumptions.amount_essential_stores * 1.0
								storeList = storesSim.essential_stores
								typeProduct = 'essential'

							else:
								N = assumptions.amount_nonessential_stores * 1.0
								storeList = storesSim.nonessential_stores
								typeProduct = 'nonessential'

							p = np.ones(int(N)) / N

							storeID_dist = stats.multinomial(1,p)
							storeID      = storeID_dist.rvs()

							j = 0
							while storeID[0, j] == 0:
								j += 1

							moneySpent = stats.norm.rvs(loc=200, scale=100)
							moneySpent = np.abs(moneySpent)

							place_where_cc_was_used = \
								storesSim.get_store_i(
									storeID     = j,
									storeType   = typeProduct,
									keyProperty ='localization'
								)

							clientsSim.confirm_transaction(
								clientID                = client_i,
								moneySpent              = moneySpent,
								typeProduct             = typeProduct,
								storeID                 = j ,
								was_a_fraud             = True,
								place_where_cc_was_used = place_where_cc_was_used,
								time                    = (day-1) * 24.0 + T_n
							)
							T_n       = next(simGenerator)
					else:
						checkTypeStore = (coin == 'essential')
						ticket    = \
							clientsSim.buy(
								store_obj                   = storesSim,
								clientID                    = client_i,
								distanceGenerator           = distanceGenerator,
								priceGenerator              = priceGenerator,
								product_to_buy_is_essential = checkTypeStore
							)

						place_where_cc_was_used = \
							storesSim.get_store_i(
								storeID     = ticket[0],
								storeType   = coin,
								keyProperty ='localization'
							)

						clientsSim.confirm_transaction(
							clientID                = client_i,
							moneySpent              = ticket[1],
							typeProduct             = coin,
							storeID                 = ticket[0] ,
							was_a_fraud             = False,
							place_where_cc_was_used = place_where_cc_was_used,
							time                    = (day-1) * 24.0 + T_n
						)

				except:
					print(f'\t day -> {day} DONE.')
					break
		#2}}}


	def print_to_csv(self, file):
		#{{{2 function scope
		filePointer = open(file, 'w')

		filePointer.write(
			'clientID,'
			'buyID,'
			'time,'
			'moneySpent,'
			'shop_accepted,'
			'was_a_fraud,'
			'store_bought_from,'
			'type_product,'
			'place_where_cc_was_used_x,'
			'place_where_cc_was_used_y'
		)
		filePointer.write('\n')
		for i in range(self.assumptions.clientsPopSize):
			client_i = self.clientsSim.get_client_i(i)

			for transaction in client_i['shopping_list']:
				mystr = "{:d},{:d},{:.8f},{:.2f},{:},{:},{:d},{:},{:.2f},{:.2f}"
				filePointer.write(
					mystr.format(
						client_i['clientID'],
						transaction['buyID'],
						transaction['time'],
						transaction['moneySpent'],
						transaction['shop_accepted'],
						transaction['was_a_fraud'],
						transaction['store_bought_from'],
						transaction['type_product'],
						transaction['place_where_cc_was_used'][0],
						transaction['place_where_cc_was_used'][1]
					)
				)
				filePointer.write('\n')
		filePointer.close()
		#2}}}


	def payDebtEndMonth(self):
		#{{{2 function scope

		clientsSim     = self.clientsSim
		assumptions    = self.assumptions

		for i in range(self.assumptions.clientsPopSize):
			client = clientsSim.get_client_i(i)

			debt = client['creditCard_limitUsed']
			wage = client['estimated_wage_per_month']

			Id = 1 * (debt > wage)

			client['creditCard_limitUsed'] = Id * (debt - wage)
		#2}}}
	#1}}}



############# TESTS #######################
#simulation = fraudSimulation(amount_of_days  = 360,
#						     clientsPopSize  = 1_000,
#                             storesPopSize   = 100,
#							 ball_radius_R   = 10_000)
#
#simulation.runSim()
#simulation.print_to_csv('sim.dat')
#
#data = pd.read_csv('sim.dat')
#data = data.sort_values(by='time')
#
#dataFraud    = data[data.was_a_fraud == True and data.shop_accepted == True]
#dataNotFraud = data[data.was_a_fraud == False and data.shop_accepted == True]
#
#plt.scatter(dataNotFraud.time.values, dataNotFraud.moneySpent.values, color='blue')
#plt.scatter(dataFraud.time.values,    dataFraud.moneySpent.values, color='red')
#plt.show()

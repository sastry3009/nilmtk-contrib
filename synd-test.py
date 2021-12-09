import warnings
warnings.filterwarnings("ignore")
from nilmtk.api import API

from nilmtk_contrib.disaggregate import Seq2Seq, BERT


synd = {
  'power': {
    'mains': ['active'],
    'appliance': ['active']
  },
  'sample_rate': 10,

  'appliances': ['electric space heater', 'clothes iron', 'dish washer', 'washing machine', 'fridge'],
  'methods': {
    
#      'WindowGRU':WindowGRU({'n_epochs':50,'batch_size':32}),
#     'RNN':RNN({'n_epochs':50,'batch_size':32}),
#      'DAE':DAE({'n_epochs':50,'batch_size':32}),
#      'Seq2Point':Seq2Point({'n_epochs':50,'batch_size':32}),
     'Seq2Seq':Seq2Seq({'n_epochs':20,'batch_size':256}),
#      'BERT':BERT({})

#      'Mean': Mean({}),
          
  },
   'train': {    
    'datasets': {
            'SynD': {
                'path': '/home/sastry/Desktop/NILM-work/SynD.h5',
				'buildings': {
				1: {
					'start_time': '2019-09-29',
					'end_time': '2020-02-07'
				},
				}
				                
			}
			}
	},
	'test': {
	'datasets': {
		'SynD': {
			'path': '/home/sastry/Desktop/NILM-work/SynD.h5',
			'buildings': {
				1: {
					'start_time': '2020-02-07',
					'end_time': '2020-03-26'
				},
			}
	}
},
        'metrics':['mae', 'f1score', 'accuracy', 'precision', 'recall']
}
}

api_res = API(synd)

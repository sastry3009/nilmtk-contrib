import warnings
warnings.filterwarnings("ignore")
from nilmtk.api import API

from nilmtk_contrib.disaggregate import Seq2Seq, BERT


redd = {
  'power': {
    'mains': ['apparent','active'],
    'appliance': ['apparent','active']
  },
  'sample_rate': 120,

  'appliances': ['fridge', 'light', 'dish washer', 'sockets', 'microwave'],
  'methods': {
    
#      'WindowGRU':WindowGRU({'n_epochs':50,'batch_size':32}),
#     'RNN':RNN({'n_epochs':50,'batch_size':32}),
#      'DAE':DAE({'n_epochs':50,'batch_size':32}),
#      'Seq2Point':Seq2Point({'n_epochs':50,'batch_size':32}),
      'Seq2Seq':Seq2Seq({'n_epochs':20,'batch_size':128}),
      'BERT':BERT({})

#      'Mean': Mean({}),
          
  },
   'train': {    
    'datasets': {
            'REDD': {
                'path': '/home/sastry/Desktop/NILM-work/low_redd.h5',
				'buildings': {
				1: {
					'start_time': '2011-04-18',
					'end_time': '2011-04-30'
				},
				}
				                
			}
			}
	},
	'test': {
	'datasets': {
		'REDD': {
			'path': '/home/sastry/Desktop/NILM-work/low_redd.h5',
			'buildings': {
				1: {
					'start_time': '2011-04-30',
					'end_time': '2011-05-24'
				},
			}
	}
},
        'metrics':['mae', 'f1score', 'accuracy', 'precision', 'recall']
}
}

api_res = API(redd)

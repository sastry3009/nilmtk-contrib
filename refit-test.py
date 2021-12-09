import warnings
warnings.filterwarnings("ignore")
from nilmtk.api import API

from nilmtk_contrib.disaggregate import Seq2Seq, BERT


refit = {
  'power': {
    'mains': ['active'],
    'appliance': ['active']
  },
  'sample_rate': 8,

  'appliances': ['washing machine', 'kettle', 'fridge freezer',  'audio system', 'dish washer'],
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
            'REFIT': {
                'path': '/home/sastry/Desktop/NILM-work/refit.h5',
				'buildings': {
				2: {
					'start_time': '2014-07-10',
					'end_time': '2014-10-14'
				},
				}
				                
			}
			}
	},
	'test': {
	'datasets': {
		'REFIT': {
			'path': '/home/sastry/Desktop/NILM-work/refit.h5',
			'buildings': {
				2: {
					'start_time': '2014-09-30',
					'end_time': '2014-10-26'
				},
			}
	}
},
        'metrics':['mae', 'f1score', 'accuracy', 'precision', 'recall']
}
}

api_res = API(refit)

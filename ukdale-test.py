import warnings
warnings.filterwarnings("ignore")
from nilmtk.api import API

from nilmtk_contrib.disaggregate import Seq2Seq, BERT


ukdale = {
  'power': {
    'mains': ['apparent','active'],
    'appliance': ['apparent','active']
  },
  'sample_rate': 6,

  'appliances': ['fridge freezer', 'dish washer'], #  'microwave', 'kettle', 'washer dryer', 'fridge freezer', 'dish washer'],
  'methods': {
    
#      'WindowGRU':WindowGRU({'n_epochs':50,'batch_size':32}),
#     'RNN':RNN({'n_epochs':50,'batch_size':32}),
#      'DAE':DAE({'n_epochs':50,'batch_size':32}),
#      'Seq2Point':Seq2Point({'n_epochs':50,'batch_size':32}),
      'Seq2Seq':Seq2Seq({'n_epochs':20,'batch_size':512}),
#      'BERT':BERT({})

#      'Mean': Mean({}),
          
  },
   'train': {    
    'datasets': {
            'UK-DALE': {
                'path': '/home/sastry/Desktop/NILM-work/ukdale.h5',
				'buildings': {
				1: {
					'start_time': '2013-04-13',
					'end_time': '2014-01-01'
				},
				}
				                
			}
			}
	},
	'test': {
	'datasets': {
		'UK-DALE': {
			'path': '/home/sastry/Desktop/NILM-work/ukdale.h5',
			'buildings': {
				1: {
					'start_time': '2014-01-01',
					'end_time': '2014-03-30'
				},
			}
	}
},
        'metrics':['mae', 'f1score', 'accuracy', 'precision', 'recall']
}
}

api_res = API(ukdale)

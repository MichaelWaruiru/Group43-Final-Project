"""
Disease information and treatment recommendations database
"""

DISEASE_INFO = {
    'Healthy': {
        'description': 'Your plant appears to be healthy! No disease detected.',
        'symptoms': ['Green, vibrant leaves', 'No spots or discoloration', 'Normal growth pattern'],
        'causes': ['Good plant care', 'Proper watering', 'Adequate nutrition'],
        'severity': 'None'
    },
    'Tomato_Early_Blight': {
        'description': 'Early blight is a common fungal disease affecting tomato plants.',
        'symptoms': ['Dark spots with concentric rings on leaves', 'Yellowing around spots', 'Leaf drop'],
        'causes': ['Alternaria solani fungus', 'Warm humid conditions', 'Poor air circulation'],
        'severity': 'Moderate'
    },
    'Tomato_Late_Blight': {
        'description': 'Late blight is a serious fungal disease that can destroy tomato crops.',
        'symptoms': ['Water-soaked spots on leaves', 'White fuzzy growth on leaf undersides', 'Fruit rot'],
        'causes': ['Phytophthora infestans', 'Cool wet weather', 'High humidity'],
        'severity': 'Severe'
    },
    'Tomato_Leaf_Mold': {
        'description': 'Leaf mold is a fungal disease common in greenhouse tomatoes.',
        'symptoms': ['Yellow spots on upper leaf surface', 'Fuzzy growth on leaf undersides', 'Leaf curling'],
        'causes': ['Passalora fulva fungus', 'High humidity', 'Poor ventilation'],
        'severity': 'Moderate'
    },
    'Tomato_Septoria_Leaf_Spot': {
        'description': 'Septoria leaf spot is a fungal disease affecting tomato foliage.',
        'symptoms': ['Small circular spots with dark borders', 'White centers with black specks', 'Leaf yellowing'],
        'causes': ['Septoria lycopersici fungus', 'Wet conditions', 'Overhead watering'],
        'severity': 'Moderate'
    },
    'Tomato_Spider_Mites': {
        'description': 'Spider mites are tiny pests that can cause significant damage.',
        'symptoms': ['Fine webbing on leaves', 'Stippled or bronzed leaves', 'Tiny moving dots'],
        'causes': ['Hot dry conditions', 'Two-spotted spider mites', 'Stress conditions'],
        'severity': 'Moderate'
    },
    'Tomato_Target_Spot': {
        'description': 'Target spot is a fungal disease creating distinctive circular lesions.',
        'symptoms': ['Circular spots with concentric rings', 'Brown to black lesions', 'Leaf drop'],
        'causes': ['Corynespora cassiicola fungus', 'Warm humid conditions', 'Poor air circulation'],
        'severity': 'Moderate'
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'A viral disease transmitted by whiteflies.',
        'symptoms': ['Upward curling of leaves', 'Yellowing of leaves', 'Stunted growth'],
        'causes': ['Tomato yellow leaf curl virus', 'Whitefly transmission', 'Infected plants nearby'],
        'severity': 'Severe'
    },
    'Tomato_Mosaic_Virus': {
        'description': 'A viral disease causing mosaic patterns on leaves.',
        'symptoms': ['Mottled light and dark green patterns', 'Distorted leaf growth', 'Reduced fruit quality'],
        'causes': ['Tomato mosaic virus', 'Mechanical transmission', 'Contaminated tools'],
        'severity': 'Moderate'
    },
    'Tomato_Bacterial_Spot': {
        'description': 'A bacterial disease causing spots on leaves and fruit.',
        'symptoms': ['Small dark spots with yellow halos', 'Spots on fruit', 'Leaf drop'],
        'causes': ['Xanthomonas bacteria', 'Warm wet conditions', 'Overhead irrigation'],
        'severity': 'Moderate'
    },
    'Potato_Early_Blight': {
        'description': 'Early blight affects potato plants, causing leaf spots and tuber damage.',
        'symptoms': ['Circular dark spots on leaves', 'Target-like patterns', 'Yellowing leaves'],
        'causes': ['Alternaria solani fungus', 'Warm humid weather', 'Plant stress'],
        'severity': 'Moderate'
    },
    'Potato_Late_Blight': {
        'description': 'Late blight is a devastating disease of potatoes.',
        'symptoms': ['Water-soaked lesions', 'White growth on leaf undersides', 'Tuber rot'],
        'causes': ['Phytophthora infestans', 'Cool wet conditions', 'High humidity'],
        'severity': 'Severe'
    },
    'Potato_Healthy': {
        'description': 'Your potato plant appears healthy!',
        'symptoms': ['Green vigorous foliage', 'No disease symptoms', 'Normal growth'],
        'causes': ['Good growing conditions', 'Proper care', 'Disease-free environment'],
        'severity': 'None'
    },
    'Corn_Common_Rust': {
        'description': 'Common rust is a fungal disease affecting corn leaves.',
        'symptoms': ['Small reddish-brown pustules', 'Oval-shaped lesions', 'Yellowing leaves'],
        'causes': ['Puccinia sorghi fungus', 'Moderate temperatures', 'High humidity'],
        'severity': 'Moderate'
    },
    'Corn_Northern_Leaf_Blight': {
        'description': 'Northern leaf blight causes large lesions on corn leaves.',
        'symptoms': ['Large gray-green lesions', 'Cigar-shaped spots', 'Leaf death'],
        'causes': ['Exserohilum turcicum fungus', 'Warm humid conditions', 'Dense plantings'],
        'severity': 'Moderate'
    },
    'Corn_Healthy': {
        'description': 'Your corn plant looks healthy!',
        'symptoms': ['Green leaves', 'No lesions or spots', 'Normal development'],
        'causes': ['Optimal growing conditions', 'Good plant health', 'Disease prevention'],
        'severity': 'None'
    },
    'Pepper_Bacterial_Spot': {
        'description': 'Bacterial spot affects pepper plants and fruit.',
        'symptoms': ['Small dark spots on leaves', 'Yellow halos around spots', 'Fruit lesions'],
        'causes': ['Xanthomonas bacteria', 'Warm wet weather', 'Overhead watering'],
        'severity': 'Moderate'
    },
    'Pepper_Healthy': {
        'description': 'Your pepper plant is healthy!',
        'symptoms': ['Vibrant green foliage', 'No disease symptoms', 'Good growth'],
        'causes': ['Proper care', 'Good environment', 'Disease-free conditions'],
        'severity': 'None'
    }
}

TREATMENT_RECOMMENDATIONS = {
    'Healthy': {
        'preventive': [
            'Continue current care practices',
            'Monitor plants regularly',
            'Maintain proper watering schedule',
            'Ensure adequate nutrition'
        ],
        'organic': [
            'Use organic fertilizers',
            'Apply compost regularly',
            'Encourage beneficial insects',
            'Practice crop rotation'
        ],
        'chemical': []
    },
    'Tomato_Early_Blight': {
        'preventive': [
            'Improve air circulation',
            'Avoid overhead watering',
            'Remove infected plant debris',
            'Rotate crops annually'
        ],
        'organic': [
            'Apply baking soda spray (1 tsp per quart water)',
            'Use copper-based fungicides',
            'Apply neem oil spray',
            'Mulch around plants'
        ],
        'chemical': [
            'Apply chlorothalonil fungicide',
            'Use mancozeb fungicide',
            'Apply azoxystrobin fungicide',
            'Follow label instructions carefully'
        ]
    },
    'Tomato_Late_Blight': {
        'preventive': [
            'Plant resistant varieties',
            'Improve air circulation',
            'Avoid overhead irrigation',
            'Remove infected plants immediately'
        ],
        'organic': [
            'Apply copper fungicide',
            'Use Bacillus subtilis biofungicide',
            'Remove and destroy infected plants',
            'Improve drainage'
        ],
        'chemical': [
            'Apply propamocarb fungicide',
            'Use dimethomorph fungicide',
            'Apply copper hydroxide',
            'Act quickly for best results'
        ]
    },
    'Tomato_Leaf_Mold': {
        'preventive': [
            'Increase ventilation',
            'Reduce humidity',
            'Space plants properly',
            'Remove lower leaves'
        ],
        'organic': [
            'Apply baking soda solution',
            'Use milk spray (1:10 ratio)',
            'Apply neem oil',
            'Improve air circulation'
        ],
        'chemical': [
            'Use chlorothalonil fungicide',
            'Apply copper fungicide',
            'Use myclobutanil fungicide',
            'Rotate fungicide types'
        ]
    },
    'Tomato_Spider_Mites': {
        'preventive': [
            'Maintain adequate humidity',
            'Avoid over-fertilization',
            'Regular plant inspection',
            'Remove weeds around plants'
        ],
        'organic': [
            'Spray with water regularly',
            'Apply neem oil spray',
            'Use predatory mites',
            'Apply insecticidal soap'
        ],
        'chemical': [
            'Apply bifenthrin miticide',
            'Use abamectin spray',
            'Apply spiromesifen',
            'Rotate miticide classes'
        ]
    },
    'Tomato_Yellow_Leaf_Curl_Virus': {
        'preventive': [
            'Control whitefly populations',
            'Remove infected plants',
            'Use row covers',
            'Plant resistant varieties'
        ],
        'organic': [
            'Use yellow sticky traps',
            'Apply neem oil for whiteflies',
            'Encourage beneficial insects',
            'Remove infected plants'
        ],
        'chemical': [
            'Apply systemic insecticides for whiteflies',
            'Use imidacloprid soil drench',
            'Apply thiamethoxam spray',
            'Remove and destroy infected plants'
        ]
    },
    'Potato_Late_Blight': {
        'preventive': [
            'Plant certified seed potatoes',
            'Improve air circulation',
            'Avoid overhead watering',
            'Hill soil around plants'
        ],
        'organic': [
            'Apply copper fungicide',
            'Use Bacillus subtilis',
            'Remove infected tubers',
            'Improve soil drainage'
        ],
        'chemical': [
            'Apply metalaxyl fungicide',
            'Use propamocarb treatment',
            'Apply dimethomorph spray',
            'Treat preventively in high-risk periods'
        ]
    },
    'Corn_Common_Rust': {
        'preventive': [
            'Plant resistant varieties',
            'Ensure good air circulation',
            'Avoid overhead irrigation',
            'Remove infected debris'
        ],
        'organic': [
            'Apply sulfur fungicide',
            'Use copper-based treatment',
            'Apply compost tea',
            'Encourage beneficial microbes'
        ],
        'chemical': [
            'Apply propiconazole fungicide',
            'Use azoxystrobin spray',
            'Apply tebuconazole treatment',
            'Time applications during early infection'
        ]
    },
    'Pepper_Bacterial_Spot': {
        'preventive': [
            'Use drip irrigation',
            'Improve air circulation',
            'Sanitize tools',
            'Remove infected plant debris'
        ],
        'organic': [
            'Apply copper fungicide',
            'Use hydrogen peroxide spray',
            'Apply baking soda solution',
            'Remove infected leaves'
        ],
        'chemical': [
            'Apply copper hydroxide',
            'Use streptomycin spray',
            'Apply copper sulfate',
            'Combine with fungicides if needed'
        ]
    }
}

def get_disease_info(disease_name):
  """Get disease information"""
  return DISEASE_INFO.get(disease_name, {
    'description': 'Disease information not available',
    'symptoms': ['Information not available'],
    'causes': ['Information not available'],
    'severity': 'Unknown'
  })

def get_treatment_recommendations(disease_name):
  """Get treatment recommendations for a disease"""
  return TREATMENT_RECOMMENDATIONS.get(disease_name, {
    'preventive': ['Consult with agricultural extension service'],
    'organic': ['Apply general organic treatments'],
    'chemical': ['Consult with agricultural specialist']
  })

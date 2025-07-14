"""
Disease information and treatment recommendations database
"""

DISEASE_INFO = {
    'Pepper__bell___Bacterial_spot': {
        'description': 'Bacterial spot affects pepper plants and fruit.',
        'symptoms': [
            'Small dark spots on leaves',
            'Yellow halos around spots',
            'Fruit lesions'
        ],
        'causes': [
            'Xanthomonas bacteria',
            'Warm wet weather',
            'Overhead watering'
        ],
        'severity': 'Moderate'
    },
    'Pepper__bell___healthy': {
        'description': 'Your pepper plant is healthy!',
        'symptoms': [
            'Vibrant green foliage',
            'No disease symptoms',
            'Good growth'
        ],
        'causes': [
            'Proper care',
            'Good environment',
            'Disease-free conditions'
        ],
        'severity': 'None'
    },
    'Potato___Early_blight': {
        'description': 'Early blight causes concentric leaf spots and stem lesions in potatoes.',
        'symptoms': [
            'Circular dark spots on leaves',
            'Target-like patterns',
            'Yellowing of foliage',
            'Reduced yield'
        ],
        'causes': [
            'Alternaria solani fungus',
            'Warm humid weather',
            'Plant stress'
        ],
        'severity': 'Moderate'
    },
    'Potato___healthy': {
        'description': 'Your potato plant appears healthy!',
        'symptoms': [
            'Green vigorous foliage',
            'No disease symptoms',
            'Normal growth'
        ],
        'causes': [
            'Good growing conditions',
            'Proper care',
            'Disease-free environment'
        ],
        'severity': 'None'
    },
    'Potato___Late_blight': {
        'description': 'Late blight can destroy potato foliage and cause tuber rot.',
        'symptoms': [
            'Dark lesions on leaves',
            'Soft rot in tubers',
            'White mold on leaf undersides in wet conditions'
        ],
        'causes': [
            'Phytophthora infestans',
            'Cool wet conditions',
            'High humidity'
        ],
        'severity': 'Severe'
    },
    'Tomato__Target_Spot': {
        'description': 'Target spot is a fungal disease creating distinctive circular lesions.',
        'symptoms': [
            'Circular spots with concentric rings',
            'Brown to black lesions',
            'Leaf drop'
        ],
        'causes': [
            'Corynespora cassiicola fungus',
            'Warm humid conditions',
            'Poor air circulation'
        ],
        'severity': 'Moderate'
    },
    'Tomato__Tomato_mosaic_virus': {
        'description': 'A viral disease causing mosaic patterns on tomato leaves.',
        'symptoms': [
            'Mottled light and dark green patterns',
            'Distorted leaf growth',
            'Reduced fruit quality'
        ],
        'causes': [
            'Tomato mosaic virus',
            'Mechanical transmission',
            'Contaminated tools'
        ],
        'severity': 'Moderate'
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'description': 'A viral disease transmitted by whiteflies.',
        'symptoms': [
            'Upward curling of leaves',
            'Yellowing of leaves',
            'Stunted growth'
        ],
        'causes': [
            'Tomato yellow leaf curl virus',
            'Whitefly transmission',
            'Infected plants nearby'
        ],
        'severity': 'Severe'
    },
    'Tomato_Bacterial_spot': {
        'description': 'A bacterial disease causing spots on tomato leaves and fruit.',
        'symptoms': [
            'Small dark spots with yellow halos',
            'Spots on fruit',
            'Leaf drop'
        ],
        'causes': [
            'Xanthomonas bacteria',
            'Warm wet conditions',
            'Overhead irrigation'
        ],
        'severity': 'Moderate'
    },
    'Tomato_Early_blight': {
        'description': 'Early blight is a common fungal disease affecting tomato plants.',
        'symptoms': [
            'Dark spots with concentric rings on leaves',
            'Yellowing around spots',
            'Leaf drop'
        ],
        'causes': [
            'Alternaria solani fungus',
            'Warm humid conditions',
            'Poor air circulation'
        ],
        'severity': 'Moderate'
    },
    'Tomato_healthy': {
        'description': 'Your tomato plant appears healthy!',
        'symptoms': [
            'Green, vibrant leaves',
            'No spots or discoloration',
            'Normal growth pattern'
        ],
        'causes': [
            'Good plant care',
            'Proper watering',
            'Adequate nutrition'
        ],
        'severity': 'None'
    },
    'Tomato_Late_blight': {
        'description': 'Late blight is a serious fungal disease that can destroy tomato crops.',
        'symptoms': [
            'Water-soaked spots on leaves',
            'White fuzzy growth on leaf undersides',
            'Fruit rot'
        ],
        'causes': [
            'Phytophthora infestans',
            'Cool wet weather',
            'High humidity'
        ],
        'severity': 'Severe'
    },
    'Tomato_Leaf_Mold': {
        'description': 'Leaf mold is a fungal disease common in greenhouse tomatoes.',
        'symptoms': [
            'Yellow spots on upper leaf surface',
            'Fuzzy growth on leaf undersides',
            'Leaf curling'
        ],
        'causes': [
            'Passalora fulva fungus',
            'High humidity',
            'Poor ventilation'
        ],
        'severity': 'Moderate'
    },
    'Tomato_Septoria_leaf_spot': {
        'description': 'Septoria leaf spot is a fungal disease affecting tomato foliage.',
        'symptoms': [
            'Small circular spots with dark borders',
            'White centers with black specks',
            'Leaf yellowing'
        ],
        'causes': [
            'Septoria lycopersici fungus',
            'Wet conditions',
            'Overhead watering'
        ],
        'severity': 'Moderate'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'description': 'Spider mites are tiny pests that can cause significant damage.',
        'symptoms': [
            'Fine webbing on leaves',
            'Stippled or bronzed leaves',
            'Tiny moving dots'
        ],
        'causes': [
            'Hot dry conditions',
            'Two-spotted spider mites',
            'Stress conditions'
        ],
        'severity': 'Moderate'
    }
}

TREATMENT_RECOMMENDATIONS = {
    'Pepper__bell___Bacterial_spot': {
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
    },
    'Pepper__bell___healthy': {
        'preventive': [
            'Continue proper care',
            'Maintain watering and nutrition balance',
            'Inspect regularly for early symptoms',
            'Avoid overcrowding'
        ],
        'organic': [
            'Apply compost and mulch',
            'Encourage beneficial insects',
            'Use organic fertilizers',
            'Rotate with legumes or corn'
        ],
        'chemical': []
    },
    'Potato___Early_blight': {
        'preventive': [
            'Rotate crops regularly',
            'Remove and destroy infected plant material',
            'Ensure proper plant spacing',
            'Avoid overhead irrigation'
        ],
        'organic': [
            'Apply neem oil spray',
            'Use copper-based fungicide',
            'Add compost to improve plant health',
            'Mulch soil to avoid splashing spores'
        ],
        'chemical': [
            'Apply mancozeb fungicide',
            'Use chlorothalonil spray',
            'Rotate with azoxystrobin',
            'Start treatment early in the season'
        ]
    },
    'Potato___healthy': {
        'preventive': [
            'Use certified seed potatoes',
            'Rotate crops regularly',
            'Maintain clean fields and tools',
            'Monitor frequently for pests and disease'
        ],
        'organic': [
            'Apply compost before planting',
            'Mulch to conserve moisture and suppress weeds',
            'Use beneficial microbes',
            'Practice intercropping with legumes'
        ],
        'chemical': []
    },
    'Potato___Late_blight': {
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
    'Tomato__Target_Spot': {
        'preventive': [
            'Ensure good air circulation',
            'Avoid overhead watering',
            'Remove affected foliage',
            'Practice crop rotation'
        ],
        'organic': [
            'Apply copper fungicide',
            'Use neem oil spray',
            'Mulch to reduce soil splash',
            'Maintain dry foliage'
        ],
        'chemical': [
            'Use chlorothalonil-based fungicide',
            'Apply azoxystrobin spray',
            'Rotate with mancozeb',
            'Begin treatment at first signs of disease'
        ]
    },
    'Tomato__Tomato_mosaic_virus': {
        'preventive': [
            'Use virus-free seeds',
            'Disinfect tools regularly',
            'Avoid smoking near plants',
            'Remove infected plants promptly'
        ],
        'organic': [
            'Apply compost teas to boost immunity',
            'Use resistant tomato varieties',
            'Enhance soil health with organic matter',
            'Sanitize hands and gloves between handling plants'
        ],
        'chemical': [
            'No effective chemical treatment for viruses',
            'Control insect vectors chemically',
            'Use insecticidal soaps for aphid control',
            'Combine with preventive measures'
        ]
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
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
    'Tomato_Bacterial_spot': {
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
    },
    'Tomato_Early_blight': {
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
    'Tomato_healthy': {
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
    'Tomato_Late_blight': {
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
    'Tomato_Septoria_leaf_spot': {
        'preventive': [
            'Improve air circulation',
            'Avoid overhead watering',
            'Remove infected leaves',
            'Practice crop rotation'
        ],
        'organic': [
            'Apply copper-based fungicide',
            'Use neem oil spray',
            'Mulch to reduce soil splash',
            'Maintain dry foliage'
        ],
        'chemical': [
            'Use chlorothalonil fungicide',
            'Apply azoxystrobin spray',
            'Rotate with mancozeb',
            'Start treatment at first signs of disease'
        ]
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
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

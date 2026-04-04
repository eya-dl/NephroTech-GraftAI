import dice_ml

transformer = joblib.load('transformer.pkl')  # your ColumnTransformer
model = joblib.load('model.pkl')

# Wrap it to ignore any extra kwargs DiCE passes
def custom_transform(data, **kwargs):          # **kwargs absorbs 'data_interface'
    return transformer.transform(data)

d = dice_ml.Data(
    dataframe=train_dataset,
    continuous_features=['age', 'c_PRA', 'dialysis_time','n_pregnant', 'age_donor','final_creatinine_donor', 'cold_ischemia_time','lenght_stay','creatinine_discharge', 'MM','time_mat','year','final_creatinine'],
    outcome_name='rejection'
)

m = dice_ml.Model(
    model=model,
    backend='sklearn',
    func=custom_transform                      
)

exp = dice_ml.Dice(d, m)

# we will take the first patient 
#[0:1] to keep the dataframe format and not convert it to a series (bach to93od 2D mich 1D sinon yi5dimch)

new_patient = X_test0.iloc[0:1].reset_index(drop=True)

# first case : stuff we can change moving forward

dice_exp1 = exp.generate_counterfactuals(
    
    query_instances=new_patient,

    total_CFs=3,                          # number of counterfactuals to generate
    desired_class="opposite",             # flip the prediction (0→1 or 1→0)

    features_to_vary=[                    # only these features can change
        'immunosuppresion',
        'convertion',
        'plasmaferesis',
        'eculizumab',
        'trat_combined'
    ]


    
   
    
)


dice_exp1.visualize_as_dataframe(show_only_changes=True)  # show only what changed


dice_exp2 = exp.generate_counterfactuals(
    query_instances=new_patient,

    total_CFs=3,                         
    desired_class="opposite",             

    features_to_vary=[   
                                    
        'dialysis_type',
        'dialysis_time',
        'donor_type',
        'age_donor',
        'donor_death',
        'final_creatinine_donor',
        'cold_ischemia_time',
        'lenght_stay',
        'induction',
        'induction_eculizumab',
        'eculizumab',
        'enxertectomy'
        
    ]


    
   
    
)


dice_exp2.visualize_as_dataframe(show_only_changes=True)  # show only what changed

# third case : first and seconf case :)

dice_exp3 = exp.generate_counterfactuals(
    query_instances=new_patient,

    total_CFs=3,                         
    desired_class="opposite",             

    features_to_vary=[    
        'immunosuppresion',
        'convertion',
        'plasmaferesis',
        'eculizumab',
        'trat_combined',             
        'dialysis_type',
        'dialysis_time',
        'donor_type',
        'age_donor',
        'donor_death',
        'final_creatinine_donor',
        'cold_ischemia_time',
        'lenght_stay',
        'induction',
        'induction_eculizumab',
        'eculizumab',
        'enxertectomy'
        
    ]


    
   
    
)


dice_exp3.visualize_as_dataframe(show_only_changes=True)  # show only what changed
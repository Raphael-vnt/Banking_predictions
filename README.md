# Banking predictions results


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recall</th>
      <th>precision</th>
      <th>f1</th>
      <th>accuracy</th>
      <th>auc_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>best_lr</th>
      <td>0.421725</td>
      <td>0.678082</td>
      <td>0.520026</td>
      <td>0.911233</td>
      <td>0.934354</td>
    </tr>
    <tr>
      <th>best_rf</th>
      <td>0.489883</td>
      <td>0.677467</td>
      <td>0.568603</td>
      <td>0.915240</td>
      <td>0.944159</td>
    </tr>
    <tr>
      <th>best_xgb</th>
      <td>0.560170</td>
      <td>0.676963</td>
      <td>0.613054</td>
      <td>0.919369</td>
      <td>0.949215</td>
    </tr>
    <tr>
      <th>best_catboost</th>
      <td>0.537806</td>
      <td>0.695592</td>
      <td>0.606607</td>
      <td>0.920461</td>
      <td>0.949933</td>
    </tr>
    <tr>
      <th>best_ann</th>
      <td>0.534611</td>
      <td>0.632242</td>
      <td>0.579342</td>
      <td>0.911475</td>
      <td>0.938765</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_names = list(best_model.keys())
plt.figure(figsize=(10, 8))

for i, (name, model) in enumerate(best_model.items()):
    y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f'{name} (AP = {average_precision_score(y_test, y_pred_proba):.2f})')

plt.legend(loc='lower left')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Precision/Recall')

plt.show()
```


    
![png](banking_predictions_files/banking_predictions_101_0.png)
    



```python
model_names = list(best_model.keys())
plt.figure(figsize=(10, 8))

for i, (name, model) in enumerate(best_model.items()):
    
    y_pred_proba = model.predict_proba(X_test_prepared)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')


plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.legend(loc='lower right')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_102_0.png)
    


### 3.2 Feature importance

Nous décidons également de regarder la feature importance de notre meilleur modèle. Pour
cela nous utilisons la méthode des permutations implémentée par Sklearn.
Cette méthode consiste à mesurer la diminution de la performance d'un modèle lorsqu'on
permute de manière aléatoire les valeurs d'une caractéristique particulière, ce qui permet de
quantifier l'impact de cette caractéristique sur les prédictions du modèle. Plus la performance
du modèle diminue après permutation, plus la caractéristique est jugée importante


```python
from sklearn.inspection import permutation_importance

num_cols = var_num_final
cat_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(var_cat_final)
all_cols = num_cols + list(cat_cols)

result = permutation_importance(
    best_model['best_rf'], X_test_prepared, y_test, n_repeats=10, random_state=42, n_jobs=2
)

```


```python
features_importances = pd.Series(result.importances_mean, index=all_cols).sort_values(ascending=False)


fig, ax = plt.subplots(figsize=(10, 6))
features_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_106_0.png)
    


Nous voyons dès lors que sur les 55 variables, beaucoup ne sont que peu significatives.
Également que la variable duration a une part importante dans l’explicabilité.

### 3.3 Calibration des modèles

Dans cette dernière partie, nous analysons la calibration du modèle. La sortie des modèles étant des scores de probabilités on attendrait d'un modèle parfaitement calibré que lorsque ce dernier prédit une probabilité de 90%,
la proportion observée soit effectivement de 90% ; et ce pour tout pourcentage.


```python
from sklearn.calibration import CalibrationDisplay

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}

for i, (name, model) in enumerate(models.items()):
    if name == 'Catboost': 
        model.fit(X_train_prepared, y_train, plot=False, logging_level='Silent')
    else:
        model.fit(X_train_prepared, y_train)
    
    display = CalibrationDisplay.from_estimator(
        model,
        X_test_prepared,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.set_title("Calibration plots for all models")

grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (name, _) in enumerate(models.items()):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()
```


    
![png](banking_predictions_files/banking_predictions_110_0.png)
    


L'ensemble des modèles sont relativement bien calibrés avec Catboost comme modèle ayant la meilleure calibration. 


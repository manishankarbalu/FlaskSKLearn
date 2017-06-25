# Flask APP for scikit-learn predictions
Flask application that can predict on any input data on specified model from a scikit-learn. Predictions for a given input is made from the latest trained checkpoint stored We can also train/retrain a new model and store it for later use.
> Any sklearn model can be used for prediction.
### Dependencies
```
pip install -r requirements.txt

```
#### Sample Input
```
 [
  {'cement': 540,  'slag':0.0 ,  'ash':0.0 , 'water':162.0 , 'superplastic':2.5 , 'coarseagg':1040.0  ,'fineagg':676.0  ,'age':28 }
 ]
```
#### Sample Output:
```
{'prediction': [0, 1, 1, 0]}
```
## Usage
Run ``main.py`` and open ``0.0.0.0:5000/`` in the browser and use any of the options below 

* /train (GET) - Trains the model.

* /predict (GET) - The Predictions are returned in json format.

* /clearModel (GET) - Deletes all the model stored in the models folder.
### ToDo
- [x] Train.
- [x] clear.
- [x] Predict.
- [x] Hardcoded testing
- [ ] Save checkpoint for individual(any number of) models.
- [ ] UI for input, output and to download checkpoint.




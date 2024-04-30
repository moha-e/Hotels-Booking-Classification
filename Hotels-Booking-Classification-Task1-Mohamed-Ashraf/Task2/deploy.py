from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

with open("H:/Cellula - Machine learning intern/Intern Tasks/project 1/Task2/rf_Classification.sav", "rb") as file:
    saved_data = pickle.load(file)
    model = saved_data["model"]
    scaler_min = float(saved_data["scaler_min"])
    scaler_max = float(saved_data["scaler_max"])

@app.route("/" )
def home():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
        room_type_dict={"Room_Type 1":0,"Room_Type 2":1,"Room_Type 3":2,"Room_Type 4":3,"Room_Type 5":4,"Room_Type 6":5,
                        "Room_Type 7":6}
        
        market_segment_type_dict={"Aviation":0,"Complementary":1,"Corporate":2,"Offline":3,"Online":4}
        
        meal_type_dict={"Meal Plan 1":0,"Meal Plan 2":1,"Meal Plan 3":2,"Not Selected":3}
        prediction=0
        if request.method == "POST":

            number_of_adults=int(request.form.get("number_of_adults"))
            number_of_children=int(request.form.get("number_of_children"))
            number_of_weekend_nights=int(request.form.get("number_of_weekend_nights"))
            number_of_week_nights=int(request.form.get("number_of_week_nights"))
            car_parking_space=int(request.form.get("car_parking_space"))
            repeated=int(request.form.get("repeated"))
            previously_reservation_canceled=int(request.form.get("previously_reservation_canceled"))
            previously_reservation_not_canceled=int(request.form.get("previously_reservation_not_canceled"))
            special_requests=int(request.form.get("special_requests"))

            type_of_meal=np.int32(meal_type_dict[request.form.get("type_of_meal")])
            room_type=np.int32(room_type_dict[request.form.get("room_type")])
            market_segment_type=np.int32(market_segment_type_dict[request.form.get("market_segment_type")])

            lead_time=float((float(request.form.get("lead_time"))- scaler_min) / (scaler_max - scaler_min))
            average_price=float((float(request.form.get("average_price"))- scaler_min) / (scaler_max - scaler_min))
            day=float((float(request.form.get("day"))- scaler_min) / (scaler_max - scaler_min))
            month=float((float(request.form.get("month"))- scaler_min) / (scaler_max - scaler_min))
            year=float((float(request.form.get("year"))- scaler_min) / (scaler_max - scaler_min))

            prediction=model.predict([[number_of_adults,number_of_children,number_of_weekend_nights,number_of_week_nights,type_of_meal,car_parking_space,room_type,lead_time,market_segment_type,repeated,previously_reservation_canceled,previously_reservation_not_canceled,average_price,special_requests,day,month,year]])[0]

            print("Prediction:", prediction)
            return render_template("index.html", prediction= prediction)  # Pass the prediction value to the template

        return render_template("index.html", prediction='')

if __name__ =="__main__":
    app.run(debug=True)

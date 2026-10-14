from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from .models import Listing, DB, add_update_listing
from.predict import predict_rate
from os import getenv


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = getenv("DATABASE_URL", "sqlite:///listings.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    DB.init_app(app)

    with app.app_context():
        DB.create_all()


    #Home page
    @app.route('/', methods = ['POST', 'GET'])
    def root():
        return render_template("test.html", title="Home", listings=Listing.query.all())
       

    #When user clicks "+ New listing" button
    @app.route('/new_listing')
    def new_listing():
        return render_template("test.html")


    #When user clicks "My listings" button
    @app.route('/listings')
    def listings():
        return render_template("listings.html", title="listings", listings=Listing.query.all())

    
    #When user clicks on the listing name link in the "My listings" page
    @app.route('/listings/<listing_name>')
    def list_features(listing_name=None):
        listing_name = listing_name or request.values["listing_name"]
        listing = Listing.query.filter(Listing.listing_name == listing_name).one()
        return render_template("listing_features.html", title=f"{listing_name}'s Features", listing_name=listing_name, listing=listing)


    #When user clicks "Delete listing" button
    @app.route('/delete_listing/<listing_name>')
    def delete_listing(listing_name=None, methods=["POST", "GET"]):
        listing_name = listing_name or request.value["listing_name"]
        listing = Listing.query.filter(Listing.listing_name == listing_name).first()
        DB.session.delete(listing)
        DB.session.commit()
        add_listing_status_message = "Listing successfully deleted!"
        return render_template("test.html", Title="Listings", add_listing_status_message = add_listing_status_message)


    #Resets the database
    @app.route('/reset')
    def reset():
        DB.drop_all()
        DB.create_all()
        return render_template("test.html", title="Reset database!")


    #When "Adjust settings" is clicked from the user's listings page
    # View populated fields
    @app.route('/adjust_listing/<listing_name>', methods=['GET'])
    def adjust_listing(listing_name=None):
        listing_name=listing_name or request.values['listing_name']

        cities = ["Boston", "New York City", "Los Angeles", "Washington DC", "Chicago", "San Francisco"]
        properties = ["Apartment", "Bed & Breakfast", "Bungalow", "Condominium", "Guest House", 
                    "Hostel", "House", "Loft", "Townhouse", "Other"]
        rooms = ["Entire House/apt", "Private room", "Shared room"]
        beds = ["Real bed", "Futon", "Pull-out sofa", "Airbed", "Couch"]
        cleaning = ["Yes", "No"]
        pic = ["Yes", "No"]
        verified = ["Yes", "No"]
        instant = ["Yes", "No"]
        cancellation = ["Flexible", "Moderate", "Strict", "Super strict - 30", "Super strict - 60"]
        listing= Listing.query.filter(Listing.listing_name == listing_name).one()
        return render_template("adjust_listing.html", title="Tweak listings", listing_name=listing_name, listing=listing, 
                                all_cities=cities, property_types = properties, room_types = rooms, bed_types = beds, cleaning_types = cleaning,
                                pic_types = pic, verified_types = verified, instant_types = instant, cancellation_types = cancellation)
    

    @app.route('/about')
    def about():
        return render_template("about.html")

    
    #When user clicks on "Add listing" or "Update listing" button
    @app.route('/add_listing', methods=['POST', 'GET'])
    def add_listing():
        previous= Listing.query.filter_by(listing_name = request.values["listing_name"]).first()
        #Check if the listing name is already in DB. If it's not, then the user is trying to add a new listing and status message should display the right message.
        if previous is None:
            add_listing_status_message="Listing successfully added!"
        #If listing name is in the DB, then the user is trying to update listing data and the status message should display the right message
        else:
            add_listing_status_message="Listing successfully updated!"
        
        add_update_listing(property_type= request.values['property_type'], room_type= request.values['room_type'], amenities= request.values['amenities'], 
            accommodates= request.values['accommodates'], bathrooms=request.values['bathrooms'], bed_type= request.values['bed_type'], 
            cancellation_policy= request.values['cancellation_policy'], cleaning_fee= request.values['cleaning_fee'], city=request.values['city'], 
                description= request.values['description'], host_has_profile_pic= request.values['host_has_profile_pic'], host_identity_verified= request.values['host_identity_verified'], 
                instant_bookable=request.values['instant_bookable'], number_of_reviews= request.values['number_of_reviews'], 
            bedrooms= request.values['bedrooms'], beds= request.values['beds'], name= request.values['listing_name'], zipcode= request.values['zipcode'])
                

        return render_template('test.html', title='Home', listings=Listing.query.all(), add_listing_status_message=add_listing_status_message)       
        #return request.values['city']


    #When user clicks "Predict rate" button
    @app.route('/predict', methods=['POST', 'GET'])
    def predict():

        add_update_listing(property_type= request.values['property_type'], room_type= request.values['room_type'], amenities= request.values['amenities'], 
            accommodates= request.values['accommodates'], bathrooms=request.values['bathrooms'], bed_type= request.values['bed_type'], 
            cancellation_policy= request.values['cancellation_policy'], cleaning_fee= request.values['cleaning_fee'], city=request.values['city'], 
                description= request.values['description'], host_has_profile_pic= request.values['host_has_profile_pic'], host_identity_verified= request.values['host_identity_verified'], 
                instant_bookable=request.values['instant_bookable'], number_of_reviews= request.values['number_of_reviews'], 
            bedrooms= request.values['bedrooms'], beds= request.values['beds'], name= request.values['listing_name'], zipcode= request.values['zipcode'])
        
        prediction = predict_rate()
        predict_message = f"Your optimal nightly rate is ${prediction}."

        cities = ["Boston", "New York City", "Los Angeles", "Washington DC", "Chicago", "San Francisco"]
        properties = ["Apartment", "Bed & Breakfast", "Bungalow", "Condominium", "Guest House", 
                    "Hostel", "House", "Loft", "Townhouse", "Other"]
        rooms = ["Entire House/apt", "Private room", "Shared room"]
        beds = ["Real bed", "Futon", "Pull-out sofa", "Airbed", "Couch"]
        cleaning = ["Yes", "No"]
        pic = ["Yes", "No"]
        verified = ["Yes", "No"]
        instant = ["Yes", "No"]
        cancellation = ["Flexible", "Moderate", "Strict", "Super strict - 30", "Super strict - 60"]
        
        
        listing_name=request.values['listing_name']
        listing= Listing.query.filter(Listing.listing_name == listing_name).one()
        
        return render_template("adjust_listing.html", title="Predict rate", listing_name=listing_name, listing=listing, 
                                all_cities=cities, property_types = properties, room_types = rooms, bed_types = beds, cleaning_types = cleaning,
                                pic_types = pic, verified_types = verified, instant_types = instant, cancellation_types = cancellation, predict_message = predict_message)
        
    return app

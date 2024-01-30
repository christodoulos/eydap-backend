from src.config import MONGO_DBNAME
import mongoengine as me


class User(me.Document):
    email = me.EmailField(required=True, unique=True, pk=True)
    firstName = me.StringField(required=True)
    lastName = me.StringField(required=True)
    name = me.StringField(required=True)
    googleId = me.StringField(required=True)
    photoUrl = me.StringField(required=True)
    provider = me.StringField(required=True, choices=["GOOGLE"], default="GOOGLE")
    isAdmin = me.BooleanField(required=True, default=False)
    isEnabled = me.BooleanField(required=True, default=False)

    meta = {"collection": "users", "db_alias": MONGO_DBNAME}
    

    def to_mongo_dict(self):
        mongo_dict = self.to_mongo().to_dict()
        mongo_dict.pop("_id")
        return mongo_dict
    

    def get_user_by_google_id(googleId):
        return User.objects(googleId=googleId).first()
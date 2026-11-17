"""Small word lists shared by the anonymiser, the bias audit and the sample-data generator.

IMPORTANT: gender-coded name lists are crude and culturally incomplete. They are used **only to audit** the search
for unfair differences (never to rank, filter or infer anything about a real candidate).
"""

from __future__ import annotations

MALE_FIRST_NAMES = [
    "Ahmed", "Ali", "Muhammad", "Usman", "Bilal", "Hassan", "Imran", "Omar", "Faisal", "Hamza",
    "Rahul", "Amit", "Ravi", "Sanjay", "Vikram", "Arjun", "Rohit", "Karan", "Suresh", "Nikhil",
    "John", "Michael", "David", "Daniel", "James", "Robert", "Thomas", "Peter", "Mark", "Andrew",
]
FEMALE_FIRST_NAMES = [
    "Ayesha", "Fatima", "Sara", "Zainab", "Hina", "Sana", "Maryam", "Amna", "Noor", "Iqra",
    "Priya", "Anjali", "Neha", "Pooja", "Sneha", "Kavita", "Meera", "Divya", "Shreya", "Ritu",
    "Emily", "Sarah", "Maria", "Anna", "Laura", "Jessica", "Rachel", "Sophie", "Olivia", "Hannah",
]
NEUTRAL_FIRST_NAMES = ["Taylor", "Jordan", "Alex", "Sam", "Chris", "Robin", "Casey", "Jamie"]
LAST_NAMES = [
    "Khan", "Malik", "Sheikh", "Qureshi", "Siddiqui", "Raza", "Butt", "Chaudhry", "Hussain", "Farooqi",
    "Sharma", "Verma", "Gupta", "Iyer", "Nair", "Reddy", "Singh", "Patel", "Mehta", "Joshi",
    "Smith", "Johnson", "Brown", "Miller", "Davis", "Wilson", "Taylor", "Clark", "Walker", "Hall",
    "Garcia", "Rossi", "Muller", "Novak", "Larsen", "Kim", "Tanaka", "Silva", "Costa", "Okafor",
]

# pronoun / title swaps for counterfactual testing (lower-case keys; case is restored)
MALE_TO_FEMALE = {"he": "she", "him": "her", "his": "her", "himself": "herself", "mr": "ms", "mr.": "ms."}
FEMALE_TO_MALE = {"she": "he", "her": "his", "hers": "his", "herself": "himself", "mrs": "mr", "ms": "mr", "miss": "mr", "mrs.": "mr.", "ms.": "mr."}

# gendered words replaced by neutral ones during anonymisation
NEUTRAL_PRONOUNS = {
    "he": "they", "she": "they", "him": "them", "her": "their", "his": "their", "hers": "theirs",
    "himself": "themselves", "herself": "themselves",
}
TITLES = {"mr", "mrs", "ms", "miss", "mx", "dr"}

# lines such as "Date of Birth: ..." that carry protected characteristics
PERSONAL_DETAIL_KEYS = [
    "date of birth", "dob", "d.o.b", "birth date", "age", "gender", "sex", "marital status", "nationality",
    "religion", "caste", "father's name", "father name", "mother's name", "mother name", "husband's name",
    "passport", "visa status", "photo", "hobbies",
]

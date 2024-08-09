import polars as pl
import re

# Annotating hard and soft news
data = pl.read_parquet('ebnerd_large/articles.parquet')
hard = ['Erhverv', 'Økonomi', 'Politik', 'National politik', 'Konflikt og krig', 'Ansættelsesforhold', 'International politik', 'Køb og salg', 'Større transportmiddel', 'Bandekriminalitet', 'Videnskab', 'Væbnet konflikt', 'Uddannelse', 'Teknologi', 'Større katastrofe', 'Bedrageri', 'Offentlig instans', 'Samfundsvidenskab og humaniora', 'Naturvidenskab', 'Bæredygtighed og klima', 'Terror', 'Forbrugerelektronik', 'Kunstig intelligens og software','Katastrofe', 'Samfund']
soft = ['Kendt', 'Livsstil', 'Underholdning', 'Kriminalitet', 'Sport', 'Begivenhed', 'Personfarlig kriminalitet', 'Fodbold', 'Sportsbegivenhed', 'Film og tv', 'Privat virksomhed', 'Erotik', 'Sundhed', 'Transportmiddel', 'Mindre ulykke', 'Musik og lyd', 'Bolig', 'Kultur', 'Partnerskab', 'Bil', 'Mikro', 'Værdier', 'Krop og velvære', 'Reality', 'Underholdningsbegivenhed', 'Personlig begivenhed', 'Mad og drikke', 'Familieliv', 'Dyr', 'Rejse', 'Cykling', 'Makro', 'Ketcher- og batsport', 'Motorsport', 'Håndbold', 'Kosmetisk behandling', 'Tendenser', 'Vejr', 'Museum og seværdighed', 'Litteratur', 'Ungdomsuddannelse', 'Offentlig transport', 'Udlejning', 'Renovering og indretning', 'Religion', 'Grundskole', 'Byliv', 'Mindre transportmiddel', 'Videregående uddannelse', 'Kunst', 'Fritid', 'Mærkedag', 'Sygdom og behandling']

def check_news_type(topic_list, type_list):
    return any(topic in type_list for topic in topic_list)

# Check all the topics and return True if at least one of them is of type soft/hard
data = data.with_columns(
    pl.col("topics").map_elements(lambda topics: check_news_type(topics, hard), return_dtype=pl.Boolean).alias("is_hard"))
data = data.with_columns(
    pl.col("topics").map_elements(lambda topics: check_news_type(topics, soft), return_dtype=pl.Boolean).alias("is_soft"))

# Annotate political actors
political_actors = {
    "Socialdemokratiet": ["Mette Frederiksen","Magnus Heunicke","Nicolai Wammen","Mattias Tesfaye","Bjørn Brandenborg","Benny Engelbrecht","Simon Kollerup","Ida Auken","Annette Lind","Peter Hummelgaard","Kaare Dybvad Bek","Jesper Petersen","Morten Bødskov","Dan Jørgensen","Astrid Krag","Christian Rabjerg Madsen","Bjarne Laustsen","Leif Lahn Jensen","Anders Kronborg","Birgitte Vind","Anne Paulin","Trine Bramsen","Jens Joel","Mogens Jensen","Malte Larsen","Kasper Roug","Flemming Møller Mortensen","Fie Hækkerup","Thomas Monberg","Ane Halsboe-Jørgensen","Pernille Rosenkrantz-Theil","Sara Emil Baaring","Mette Gjerskov","Thomas Jensen","Frederik Vad","Rasmus Horn Langhoff","Rasmus Stoklund","Camilla Fabricius","Jeppe Bruus","Matilde Powers","Henrik Møller","Kris Jensen Skriver","Thomas Skriver Jensen","Lea Wermelin","Mette Reissmann","Kasper Sand Kjær","Maria Durhuus","Kim Aas","Per Husted","Rasmus Prehn"],
    "Danmarksdemokraterne": ["Inger Støjberg","Dennis Flydtkjær","Peter Skaarup","Søren Espersen","Karina Adsbøl","Hans Kristian Skibby","Jens Henrik Thulesen Dahl","Betina Kastbjerg","Marlene Harpsøe","Susie Jessen","Kenneth Fredslund Pedersen","Charlotte Munch","Lise Bech","Kristian Bøgsted"],
    "De Radikale": ["Martin Lidegaard","Samira Nawa","Katrine Robsøe","Lotte Rod","Zenia Stampe","Sofie Carsten Nielsen","Christian Friis Bach"],
    "Det Konservative Folkeparti": ["Søren Pape Poulsen","Mette Abildgaard","Rasmus Jarlov","Mai Mercado","Mona Juul","Helle Bonnesen","Niels Flemming Hansen","Per Larsen","Brigitte Klintskov Jerkel","Lise Bertelsen"],
    "Nye Borgerlige": ["Pernille Vermund","Lars Boje Mathiesen","Kim Edberg Andersen","Mette Thiesen","Peter Seier Christensen","Mikkel Bjørn"],
    "Socialistisk Folkeparti": ["Jacob Mark","Pia Olsen Dyhr","Kirsten Normann Andersen","Signe Munk","Karsten Hønge","Karina Lorentzen Dehnhardt","Theresa Berg Andersen","Charlotte Broman Mølbak","Sigurd Agersnap","Marianne Bigum","Carl Valentin","Sofie Lippert","Lisbeth Bech-Nielsen","Astrid Carø","Anne Valentina Berthelsen"],
    "Liberal Alliance": ["Alex Vanopslagh","Henrik Dahl","Ole Birk Olesen","Sólbjørg Jakobsen","Lars-Christian Brask","Katrine Daugaard","Steffen Frølund","Carsten Bach","Alexander Ryle","Jens Meilvang","Steffen Larsen","Helena Artmann Andersen","Louise Brown","Sandra Skalvig"],
    "Community of the People": ["Aaja Chemnitz Larsen"],
    "Javnaðarflokkurin": ["Sjúrður Skaale"],
    "Moderaterne": ["Lars Løkke Rasmussen","Henrik Frandsen","Rosa Eriksen","Jakob Engel-Schmidt","Tobias Elmstrøm","Monika Rubin","Karin Liltorp","Mette Kierkgaard","Jeppe Søe","Nanna Gotfredsen","Charlotte Bagge Hansen","Rasmus Lund-Nielsen","Kristian Klarskov","Jon Stephensen","Peter Have","Mike Fonseca"],
    "Dansk Folkeparti": ["Morten Messerschmidt","Pia Kjærsgaard","Peter Kofod","Alex Ahrendtsen","Nick Zimmer"],
    "Forward": ["Aki-Matilda Høegh-Dam"],
    "Union Party": ["Anna Falkenberg"],
    "Venstre": ["Jakob Ellemann-Jensen","Søren Gade","Sophie Løhde","Preben Bang Henriksen","Marie Bjerre","Anni Matthiesen","Karen Ellemann","Thomas Danielsen","Mads Fuglede","Erling Bonnesen","Christoffer Aagaard Melson","Jacob Jensen","Michael Aastrup Jensen","Morten Dahlin","Torsten Schack Pedersen","Hans Christian Schmidt","Lars Christian Lilleholt","Louise Schack Elholm","Troels Lund Poulsen","Jan E. Jørgensen","Hans Andersen","Peter Juel-Jensen","Linea Søgaard-Lidell"],
    "Enhedslisten": ["Pelle Dragsted","Mai Villadsen","Rosa Lund","Victoria Velásquez","Peder Hvelplund","Søren Søndergaard","Trine Mach","Jette Gottlieb","Søren Egge Rasmussen"],
    "Alternativet": ["Franciska Rosenkilde","Christina Olumeko","Torsten Gejl","Helene Liliendahl Brydensholt","Sasha Faxe","Theresa Scavenius"]
}

def count_pattern(text, pattern):
    return len(re.findall(pattern, text))

# if the article body contains a mention of one of its representatives, return True
for party in political_actors.keys():
    people_from_party = political_actors.get(party, [])
    pattern = '|'.join(re.escape(person) for person in people_from_party)
    data = data.with_columns(
        pl.col('body').map_elements(lambda text: count_pattern(text, pattern), return_dtype=pl.Int8).alias(party)
    )

data.write_parquet('data/ebnerd_large/articles_augmented.parquet')

import pandas as pd

path = '../namesbystate/'
state = 'merged'
full_path = path+state+'.csv'
data_set = pd.read_csv(full_path)
criteria = "Occurence"

# OR → criteria = "Percent"

# REM : f(a|b, c) means f(a) knowing b and c (e.g. with b and c fixed)

# INPUT : State & Name
# OUTPUT : Occurence = f(Year|State, Name)
def get_year(state, name, data_set_= data_set):
    return data_set_[data_set_["State"]==state][data_set_["Name"]==name]

# INPUT : Name
# OUTPUT : for each State : Occurence = f(year|State, Name)
def get_year_list(name, data_set_= data_set):
    return data_set_[data_set_["Name"]==name].pivot_table(index=["Year"],columns='State',values="Occurence")

# INPUT : Name
# OUTPUT : On average for the USA : Occurence = f(year|Name)
def get_year_USA(name, data_set_=data_set):
    if criteria=="Percent" : print("/!\ get_year_USA USES OCCURENCE, NOT PERCENT /!\ ")
    return data_set_[data_set_["Name"]==name].pivot_table(index=["Year"],columns='State',values="Occurence").sum(axis=1)

    
# INPUT : Year & Name
# OUTPUT : Occurence = f(State|Year, Name)    
def get_state(year, name, data_set_= data_set):
    return data_set_[data_set_["Year"]==year][data_set_["Name"]==name]

# INPUT : Year & State
# OUTPUT : Occurence = f(Name|Year, State)
def get_name(year, state, data_set_= data_set):
    return data_set_[data_set_["Year"]==year][data_set_["State"]==state]


# OUTPUT : List of every states in the data set
def get_list_states(data_set_ = data_set):
    return data_set["State"].drop_duplicates()


# INPUT (optional) unique = False => ith name is the most popular in the (1910+i)th year
#       (optional) period = [a,b] => only select the popular names from this period
# OUTPUT : For every year of the period, returns the most popular among the states with the number of occurences
# FORMAT : LIST
def get_list_most_used_names_year(unique = True, data_set_ = data_set):
    temp_val = data_set_.pivot_table(index="Year", columns=["Name","State"],values="Occurence").idxmax(axis=1).values
    name_list = [name for (name, state) in temp_val]
    if not unique:
        return name_list
    return [name_list[i] for i in sorted(np.unique(name_list, return_index=True)[1])]

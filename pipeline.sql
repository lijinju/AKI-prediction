

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e6dc10f3-82c9-4ddd-8496-cc69f3756d25"),
    Demographic_for_tableone=Input(rid="ri.foundry.main.dataset.77e107cc-6218-4565-86a9-754d545a72bd")
)
SELECT  has_AKI AS outcome, age_num AS age, CASE WHEN group_id = 1 then 1 else 2 end as vaccine_infection, ---previous: vaccine ,1;infection 0;now vaccine 1 infection 2
CASE WHEN gender = 'M' THEN 1 ELSE 0 END AS gender,
CASE WHEN race = 'white' THEN 0
   WHEN race = 'black' THEN 1
   WHEN race = 'no info' THEN 2
   WHEN race = 'asian' THEN 3
   WHEN race = 'other' THEN 4 ELSE 5 END AS race,
CASE WHEN ethnicity_1 = 'Not Hispanic or Latino' THEN 0
   WHEN ethnicity_1 = 'Hispanic or Latino' THEN 1
   WHEN race = 'UNKNOWN' THEN 2 ELSE 3 END AS ethnicity,
past_AKI, hypertension,diabetes_mellitus, heart_failure,cardiovascular_disease,obesity
FROM Demographic_for_tableone ;

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.877ff7a6-ff31-4f60-a5f3-b28978fa253d"),
    Demographic_for_tableone=Input(rid="ri.foundry.main.dataset.77e107cc-6218-4565-86a9-754d545a72bd")
)
SELECT  person_id,has_AKI AS outcome,
CASE WHEN gender = 'M' THEN 1 ELSE 0 END AS gender,
CASE WHEN race = 'white' THEN 0
   WHEN race = 'black' THEN 1
   WHEN race = 'no info' THEN 2
   WHEN race = 'asian' THEN 3
   WHEN race = 'other' THEN 4 ELSE 5 END AS race,
CASE WHEN ethnicity_1 = 'Not Hispanic or Latino' THEN 0
   WHEN ethnicity_1 = 'Hispanic or Latino' THEN 1
   WHEN race = 'UNKNOWN' THEN 2 ELSE 3 END AS ethnicity,
past_AKI, hypertension,diabetes_mellitus, heart_failure,cardiovascular_disease,obesity
FROM Demographic_for_tableone
WHERE group_id = 1;

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.83ec449a-947d-44d8-83a9-88e9743f0ab8"),
    Demographic_for_tableone=Input(rid="ri.foundry.main.dataset.77e107cc-6218-4565-86a9-754d545a72bd")
)
SELECT  has_AKI AS outcome, age_num AS age, CASE WHEN group_id = 1 then 1 else 0 end as vaccine,
CASE WHEN gender = 'M' THEN 1 ELSE 0 END AS gender,
CASE WHEN race = 'white' THEN 0
   WHEN race = 'black' THEN 1
   WHEN race = 'no info' THEN 2
   WHEN race = 'asian' THEN 3
   WHEN race = 'other' THEN 4 ELSE 5 END AS race,
CASE WHEN ethnicity_1 = 'Not Hispanic or Latino' THEN 0
   WHEN ethnicity_1 = 'Hispanic or Latino' THEN 1
   WHEN race = 'UNKNOWN' THEN 2 ELSE 3 END AS ethnicity,
past_AKI, hypertension,diabetes_mellitus, heart_failure,cardiovascular_disease,obesity
FROM Demographic_for_tableone 
WHERE group_id=1;


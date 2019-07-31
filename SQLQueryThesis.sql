
use thesis;







create table student_info(
id INT,
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
gender nvarchar(50),
imd_band nvarchar(50),
highest_education nvarchar(50),
age_band nvarchar(50),
num_of_prev_attempts nvarchar(50),
studied_credits nvarchar(50),
region nvarchar(50),
disability nvarchar(50),
final_result nvarchar(50)	 
);

--drop table student_Info;

select * from studentinfo; --32593


select distinct id_student from student_Info --28785
select distinct id_student from New_studentinfo

insert into student_info
SELECT * FROM studentinfo
WHERE id IN
(SELECT MIN(id) FROM studentinfo GROUP BY id_student)

select * from student_info ---28785


Select * from student_Info where code_presentation = '2014J'; --11260

Select * from student_Info where code_presentation = '2014B'; --7804

Select * from student_Info where code_presentation = '2013J';	--8845

Select * from student_Info where code_presentation = '2013B'; --4684

select distinct * from student_Info;

Select distinct id_student from student_Info where code_presentation = '2014J'; --11260

--student info 28785--

select * from studentRegistration
select distinct id_student from studentRegistration

CREATE TABLE student_Registration(
id INT IDENTITY (1,1),
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
date_registration INT,
date_unregistration INT
);

insert into student_Registration
select * from studentRegistration

drop table studentRegistration


CREATE TABLE studentRegistration(
id INT,
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
date_registration INT,
date_unregistration INT
);

insert into studentRegistration
SELECT * FROM student_Registration
WHERE id IN
(SELECT MIN(id) FROM student_Registration GROUP BY id_student)


select * from student_Registration --32593
select * from studentRegistration --28785


insert into mergeInfo1
select a.code_module, 
a.code_presentation,
a.id_student,
a.gender,
a.imd_band,
a.highest_education,
a.age_band,
a.num_of_prev_attempts,
a.studied_credits,
a.region,
a.disability,
a.final_result,
b.date_registration,
b.date_unregistration
FROM student_info a
JOIN studentRegistration b
ON a.id_student=b.id_student


create table mergeInfo1(
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
gender nvarchar(50),
imd_band nvarchar(50),
highest_education nvarchar(50),
age_band nvarchar(50),
num_of_prev_attempts nvarchar(50),
studied_credits nvarchar(50),
region nvarchar(50),
disability nvarchar(50),
final_result nvarchar(50),
date_registration INT,
date_unregistration INT
)


select count(*) from mergeInfo1; --28785

select * from studentVle --10655280

ALTER TABLE studentVle
ALTER COLUMN sum_click int;

insert into mergeclick --26074
select 
id_student,
sum(sum_click) as sum_click
from studentVle
group by id_student

create table mergeclick
(
id_student nvarchar(50),
sum_click int
)


create table mergeInfo2(
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
gender nvarchar(50),
imd_band nvarchar(50),
highest_education nvarchar(50),
age_band nvarchar(50),
num_of_prev_attempts nvarchar(50),
studied_credits nvarchar(50),
region nvarchar(50),
disability nvarchar(50),
final_result nvarchar(50),
date_registration INT,
date_unregistration INT,
sum_click INT
)

insert into mergeInfo2
select a.code_module, 
a.code_presentation,
a.id_student,
a.gender,
a.imd_band,
a.highest_education,
a.age_band,
a.num_of_prev_attempts,
a.studied_credits,
a.region,
a.disability,
a.final_result,
a.date_registration,
a.date_unregistration,
b.sum_click
FROM mergeInfo1 a
LEFT JOIN mergeclick b
ON a.id_student=b.id_student



select * from mergeInfo2 --28785

select * from studentAssessment --173912 apply right join
insert into mergeInfo3
select a.code_module, 
a.code_presentation,
a.id_student,
a.gender,
a.imd_band,
a.highest_education,
a.age_band,
a.num_of_prev_attempts,
a.studied_credits,
a.region,
a.disability,
a.final_result,
a.date_registration,
a.date_unregistration,
a.sum_click,
b.id_assessment,
b.date_submitted,
b.is_banked,
b.score
FROM mergeInfo2 a
RIGHT JOIN studentAssessment b
ON a.id_student=b.id_student

create table mergeInfo3
(
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
gender nvarchar(50),
imd_band nvarchar(50),
highest_education nvarchar(50),
age_band nvarchar(50),
num_of_prev_attempts nvarchar(50),
studied_credits nvarchar(50),
region nvarchar(50),
disability nvarchar(50),
final_result nvarchar(50),
date_registration INT,
date_unregistration INT,
sum_click INT,
id_assessment nvarchar(50),
date_submitted nvarchar(50),
is_banked nvarchar(50),
score nvarchar(50)
)


select count(*) from mergeInfo3 --173912

select * from assessments

create table mergeinfo4(
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
gender nvarchar(50),
imd_band nvarchar(50),
highest_education nvarchar(50),
age_band nvarchar(50),
num_of_prev_attempts nvarchar(50),
studied_credits nvarchar(50),
region nvarchar(50),
disability nvarchar(50),
final_result nvarchar(50),
date_registration INT,
date_unregistration INT,
sum_click INT,
id_assessment nvarchar(50),
date_submitted nvarchar(50),
is_banked nvarchar(50),
score nvarchar(50),
assessment_type nvarchar(50),
date nvarchar(50),
weight nvarchar(50)
)

insert into mergeinfo4
select a.code_module, 
a.code_presentation,
a.id_student,
a.gender,
a.imd_band,
a.highest_education,
a.age_band,
a.num_of_prev_attempts,
a.studied_credits,
a.region,
a.disability,
a.final_result,
a.date_registration,
a.date_unregistration,
a.sum_click,
a.id_assessment,
a.date_submitted,
a.is_banked,
a.score,
b.assessment_type,
b.date,
b.weight
FROM mergeInfo3 a
LEFT JOIN assessments b
ON a.id_assessment=b.id_assessment


select * from mergeinfo4 -- 173912

select * from courses --22

create table FinalStudentTable(
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
gender nvarchar(50),
imd_band nvarchar(50),
highest_education nvarchar(50),
age_band nvarchar(50),
num_of_prev_attempts nvarchar(50),
studied_credits nvarchar(50),
region nvarchar(50),
disability nvarchar(50),
final_result nvarchar(50),
date_registration INT,
date_unregistration INT,
sum_click INT,
id_assessment nvarchar(50),
date_submitted nvarchar(50),
is_banked nvarchar(50),
score nvarchar(50),
assessment_type nvarchar(50),
date nvarchar(50),
weight nvarchar(50),
module_presentation_length nvarchar(50)
)



insert into FinalStudentTable
select a.code_module, 
a.code_presentation,
a.id_student,
a.gender,
a.imd_band,
a.highest_education,
a.age_band,
a.num_of_prev_attempts,
a.studied_credits,
a.region,
a.disability,
a.final_result,
a.date_registration,
a.date_unregistration,
a.sum_click,
a.id_assessment,
a.date_submitted,
a.is_banked,
a.score,
a.assessment_type,
a.date,
a.weight,
b.module_presentation_length
from mergeinfo4 a
LEFT JOIN courses b
ON a.code_module=b.code_module
AND a.code_presentation=b.code_presentation

select * from FinalStudentTable -- 173912

select distinct id_student from FinalStudentTable				--11763
where code_module = 'BBB'
AND code_presentation = '2013B'
AND assessment_type = 'TMA'

select distinct id_student from FinalStudentTable
where code_module = 'BBB'
AND code_presentation = '2013B'







create table StudentTable_CCC2014B(
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
gender nvarchar(50),
imd_band nvarchar(50),
highest_education nvarchar(50),
age_band nvarchar(50),
num_of_prev_attempts nvarchar(50),
studied_credits nvarchar(50),
region nvarchar(50),
disability nvarchar(50),
final_result nvarchar(50),
date_registration INT,
date_unregistration INT,
sum_click INT,
id_assessment nvarchar(50),
date_submitted nvarchar(50),
is_banked nvarchar(50),
score nvarchar(50),
assessment_type nvarchar(50),
date nvarchar(50),
weight nvarchar(50),
module_presentation_length nvarchar(50)
)

insert into StudentsCCC
select * from FinalStudentTable
where code_module='CCC' AND code_presentation = '2014B' and assessment_type = 'Exam' 


select distinct sub.id_student from (
select * from FinalStudentTable where code_module='CCC' AND code_presentation = '2014B' and assessment_type = 'Exam' 
) sub

select * from StudentsCCC


select distinct sub.id_student from (
select * from FinalStudentTable where assessment_type = 'TMA' 
) sub

drop table StudentsTMA
drop table StudentTable_CCC2014B

create table StudentsCCC(
code_module nvarchar(50),
code_presentation nvarchar(50),
id_student nvarchar(50),
gender nvarchar(50),
region nvarchar(50),
highest_education nvarchar(50),
imd_band nvarchar(50),
age_band nvarchar(50),
num_of_prev_attempts nvarchar(50),
studied_credits nvarchar(50),
disability nvarchar(50),
final_result nvarchar(50),
date_registration INT,
date_unregistration INT,
sum_click INT,
id_assessment nvarchar(50),
date_submitted nvarchar(50),
is_banked nvarchar(50),
score nvarchar(50),
assessment_type nvarchar(50),
date nvarchar(50),
weight nvarchar(50),
module_presentation_length nvarchar(50)
)

insert into StudentsTMA
select * from FinalStudentTable where assessment_type = 'TMA' 


select * from StudentsTMA --98426

select * from FinalStudentTable
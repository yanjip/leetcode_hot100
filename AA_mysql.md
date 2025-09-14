```mysql
# time: 2025/9/12 21:04
# author: YanJP
# 建表
create database my_bd;
show databases;
use my_bd;
drop database my_bd;

create Table students (
    id int primary key auto_increment,
    name varchar(20) not null,
    age int,
    gender varchar(10) default 'not known'
);
show tables;
desc students;
alter table students add column class varchar(10);
alter table students modify column class varchar(10);
alter table students drop column gender;
alter table students rename to students_new;

# 查询
select * from table
select id, name, age from table

select * from table where age>20
select * from table where name!='jimy' and age>20
select * from table where age>20 or class = '1'

select * from table where name like '%wang%'
select * from table where name like '_kong'

select * from table where age between 120 and 20
select * from table where age in (20, 30, 40)

select * from table order by age ASC;

select * from table order by age ASC, score DESC;

select * from table order by score DESC LIMIT 4;

select * from table LIMIT 10,10;

select count(*) from table;
select count(*) from table where class="1"
select avg(score) from table where class="1"; # 计算 '一班'学生的平均分

# GROUP BY 通常与聚合函数一起使用，根据一个或多个列对结果集进行分组。
# HAVING 用于在 GROUP BY 分组后对结果进行过滤（WHERE 是在分组前过滤）。
select class, count(*) from table group by class;
# 查询人数大于30的班级
select class, count(*) from table group by class having count(*)>30;
select class, avg(score) from talble group by class having avg(score)>80;

# Insert
insert into table values(1, 'jimy', 201);
insert into table (name, age, class) values
("wang1", 20, "1"),
("wang2", 21, "2");

# update
update table set age=20 where name='jimy';
update table set score=score+5;

# delete
delete from table where name='jimy';

# SELECT ... FROM ... WHERE ... (查什么 从哪里 在什么条件下)
# INSERT INTO ... VALUES ... (插入到哪里 值是什么)
# UPDATE ... SET ... WHERE ... (更新哪个表 设置什么新值 在什么条件下)
# DELETE FROM ... WHERE ... (从哪里删 在什么条件下)

```
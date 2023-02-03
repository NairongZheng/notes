%% 
clear;clc;

%% 打开文件
buildings=imread('D:\code_matlab\play_play\osm_buildings_sub_area1.tif');
landuse=imread('D:\code_matlab\play_play\osm_landuse_sub_area1.tif',1);
water=imread('D:\code_matlab\play_play\osm_water_sub_area1.tif',1);

%% 前处理

% buildings不用处理

% landuse
% sub1处理完之后是0-19，10,13,14,16没有。别的区域不知道是不是，要检查一下。
landuse=landuse-min(landuse(landuse>0))+1;
landuse(landuse<0)=0;
landuse=uint8(landuse);

% water
% sub1处理完之后是0, 1, 3, 22。别的区域不知道是不是，要检查一下。
water=water-min(water(water>0))+1;
water(water<0)=0;
water=uint8(water);

%% 保存
save('buildings.mat', 'buildings');
save('landuse.mat', 'landuse');
save('water.mat', 'water');

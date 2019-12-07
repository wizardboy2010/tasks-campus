clearvars;

tiger = imread('Tiger.jpg');
deer = imread('Deer.jpg');

figure;
title('In various chroma');
subplot(231);
imshow(tiger);
title('RGB');
subplot(232);
imshow(rgb2xyz(tiger));
title('XYZ');
subplot(233);
imshow(rgb2hsv(tiger));
title('HSV');
subplot(234);
imshow(deer);
title('RGB');
subplot(235);
imshow(rgb2xyz(deer));
title('XYZ');
subplot(236);
imshow(rgb2hsv(deer));
title('HSV');

deer_hsv = rgb2hsv(deer);
tiger_hsv = rgb2hsv(tiger);

deer_maxsat = deer_hsv;
deer_maxsat(:,:,3) = 1;
tiger_maxsat = tiger_hsv;
tiger_maxsat(:,:,3) = 1;

figure;
title('Maximum Saturation');
subplot(221);
imshow(deer);
title('Deer Original');
subplot(222);
imshow(hsv2rgb(deer_maxsat));
title('Deer Maximum Saturation');
subplot(223);
imshow(tiger);
title('Tiger Original');
subplot(224);
imshow(hsv2rgb(tiger_maxsat));
title('Tiger Maximum Saturation');


deer_low = deer_hsv;
deer_low(:,:,2) = deer_hsv(:,:,2)*0.5;

tiger_low = tiger_hsv;
tiger_low(:,:,2) = tiger_hsv(:,:,2)*0.5;

figure;
title('Low Saturation');
subplot(223);
imshow(hsv2rgb(deer_low));
title('Deer Low sat')
subplot(224);
imshow(deer);
title('Deer Original')
subplot(221);
imshow(hsv2rgb(tiger_low));
title('Tiger Low sat')
subplot(222);
imshow(tiger);
title('Tiger original')


deer_high = deer_hsv;
deer_high(:,:,2) = deer_hsv(:,:,2)*1.5;

deer_low = deer_hsv;
deer_low(:,:,2) = deer_hsv(:,:,2)*0.5;

tiger_high = tiger_hsv;
tiger_high(:,:,2) = tiger_hsv(:,:,2)*1.5;

tiger_low = tiger_hsv;
tiger_low(:,:,2) = tiger_hsv(:,:,2)*0.5;


amount = 0.3;

deer_sat = deer_hsv;
deer_desat = deer_hsv;

temp = deer_hsv(:,:,3);
temp = temp+amount;
temp(temp>1) = 1;
deer_sat(:,:,3) = temp;

temp = deer_hsv(:,:,3);
temp = temp-amount;
temp(temp<0) = 0;
deer_desat(:,:,3) = temp;

tiger_sat = tiger_hsv;
tiger_desat = tiger_hsv;

temp = tiger_hsv(:,:,3);
temp = temp+amount;
temp(temp>1) = 1;
tiger_sat(:,:,3) = temp;

temp = tiger_hsv(:,:,3);
temp = temp-amount;
temp(temp<0) = 0;
tiger_desat(:,:,3) = temp;

figure;
title('Saturation-Desaturation');
subplot(234);
imshow(hsv2rgb(deer_desat));
title('Deer Low sat')
subplot(235);
imshow(deer);
title('Deer Original')
subplot(236);
imshow(hsv2rgb(deer_sat));
title('Deer High sat')
subplot(231);
imshow(hsv2rgb(tiger_desat));
title('Tiger Low sat')
subplot(232);
imshow(tiger);
title('Tiger original')
subplot(233);
imshow(hsv2rgb(tiger_sat));
title('Tiger High sat')


% data = xlsread('ciexyz31_1.csv');
% 
% x = data(:,2);
% y = data(:,3);
% z = data(:,4);
% 
% figure;
% plotChromaticity();
% hold on;
% plot(x/(x+y+z),y/(x+y+z));
% title('CIE ')

M = [[0.6067,0.1736,0.2001];[0.2988,0.5868,0.1143];[0.0000,0.0661,1.1149]];

deer_temp = [reshape(deer(:,:,1),[],1), reshape(deer(:,:,2),[],1), reshape(deer(:,:,3),[],1)];
deer_XYZ = M*double(deer_temp');

deer_max_rgb = hsv2rgb(deer_maxsat);
deer_max_temp = [reshape(deer_max_rgb(:,:,1),[],1), reshape(deer_max_rgb(:,:,2),[],1), reshape(deer_max_rgb(:,:,3),[],1)];
deer_max_XYZ = M*double(deer_max_temp');

tiger_max_rgb = hsv2rgb(tiger_maxsat);
tiger_max_temp = [reshape(tiger_max_rgb(:,:,1),[],1), reshape(tiger_max_rgb(:,:,2),[],1), reshape(tiger_max_rgb(:,:,3),[],1)];
tiger_max_XYZ = M*double(tiger_max_temp');

tiger_temp = [reshape(tiger(:,:,1),[],1), reshape(tiger(:,:,2),[],1), reshape(tiger(:,:,3),[],1)];
tiger_XYZ = M*double(tiger_temp');

deer_desat_rgb = hsv2rgb(deer_desat);
deer_desat_temp = [reshape(deer_desat_rgb(:,:,1),[],1), reshape(deer_desat_rgb(:,:,2),[],1), reshape(deer_desat_rgb(:,:,3),[],1)];
deer_desat_XYZ = M*double(deer_desat_temp');

deer_sat_rgb = hsv2rgb(deer_sat);
deer_sat_temp = [reshape(deer_sat_rgb(:,:,1),[],1), reshape(deer_sat_rgb(:,:,2),[],1), reshape(deer_sat_rgb(:,:,3),[],1)];
deer_sat_XYZ = M*double(deer_sat_temp');

tiger_desat_rgb = hsv2rgb(tiger_desat);
tiger_desat_temp = [reshape(tiger_desat_rgb(:,:,1),[],1), reshape(tiger_desat_rgb(:,:,2),[],1), reshape(tiger_desat_rgb(:,:,3),[],1)];
tiger_desat_XYZ = M*double(tiger_desat_temp');

tiger_sat_rgb = hsv2rgb(tiger_sat);
tiger_sat_temp = [reshape(tiger_sat_rgb(:,:,1),[],1), reshape(tiger_sat_rgb(:,:,2),[],1), reshape(tiger_sat_rgb(:,:,3),[],1)];
tiger_sat_XYZ = M*double(tiger_sat_temp');

for i = 1:465920
    temp = deer_XYZ(:,i);
    deer_XYZ(:,i) = temp/sum(temp);
    
    temp = deer_max_XYZ(:,i);
    deer_max_XYZ(:,i) = temp/sum(temp);
    
    temp = deer_desat_XYZ(:,i);
    deer_desat_XYZ(:,i) = temp/sum(temp);
    
    temp = deer_sat_XYZ(:,i);
    deer_sat_XYZ(:,i) = temp/sum(temp);
    
    temp = tiger_XYZ(:,i);
    tiger_XYZ(:,i) = temp/sum(temp);
    
    temp = tiger_desat_XYZ(:,i);
    tiger_desat_XYZ(:,i) = temp/sum(temp);
    
    temp = tiger_sat_XYZ(:,i);
    tiger_sat_XYZ(:,i) = temp/sum(temp);
end

figure;
title('Chromaticity');
subplot(131);
scatter(deer_desat_XYZ(1,:),deer_desat_XYZ(2,:));
title('Deer desaturated Chromaticity');
subplot(132);
scatter(deer_XYZ(1,:),deer_XYZ(2,:));
title('Deer Chromaticity');
subplot(133);
scatter(deer_sat_XYZ(1,:),deer_sat_XYZ(2,:));
title('Deer saturated Chromaticity');


figure;
title('Chromaticity');
subplot(131);
scatter(tiger_desat_XYZ(1,:),tiger_desat_XYZ(2,:));
title('tiger desaturated Chromaticity');
subplot(132);
scatter(tiger_XYZ(1,:),tiger_XYZ(2,:));
title('tiger Chromaticity');
subplot(133);
scatter(tiger_sat_XYZ(1,:),tiger_sat_XYZ(2,:));
title('tiger saturated Chromaticity');

figure;
title('Max Saturation');
subplot(121);
scatter(deer_max_XYZ(1,:),deer_max_XYZ(2,:));
title('Deer Max Sat')
subplot(122);
scatter(tiger_max_XYZ(1,:),tiger_max_XYZ(2,:));
title('TIger Max Sat')

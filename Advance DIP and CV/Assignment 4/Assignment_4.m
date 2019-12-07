clear all;

image1 = imread('AF1.jpg');
image2 = imread('AF2.jpg');

shape = size(image1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 1
image1_grey = rgb2gray(image1);
image2_grey = rgb2gray(image2);

points1 = detectSURFFeatures(image1_grey);
points2 = detectSURFFeatures(image2_grey);

[features1,valid_points1] = extractFeatures(image1_grey,points1);
[features2,valid_points2] = extractFeatures(image2_grey,points2);

%%%%%%%%%%% Displaying Features
figure;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
subplot('Position', [0 .1 .5 1]);
imshow(image1);
hold on;
mainfeat1 = valid_points1.selectStrongest(25);
mainfeat1.plot('showOrientation',true);
% points1(1:200).plot('showOrientation',true);

subplot('Position',[0.5 .1 .5 1]);
imshow(image2);
hold on;
mainfeat2 = valid_points2.selectStrongest(25);
mainfeat2.plot('showOrientation',true);
% points2(1:200).plot('showOrientation',true);

%%%%%%%%%%%%%% Matched features
indexPairs = matchFeatures(features1,features2) ;
matchedfeatures1 = valid_points1(indexPairs(:,1));
matchedfeatures2 = valid_points2(indexPairs(:,2));

figure; 
showMatchedFeatures(image1,image2,matchedfeatures1,matchedfeatures2);
legend('matched points 1','matched points 2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 2

[fund_ransac, points_used] = estimateFundamentalMatrix(matchedfeatures1, matchedfeatures2, 'Method', 'ransac','NumTrials',2000);

[ep1, ~, ep2] = svd(fund_ransac);

%%% Non Homogeneous
ep1 = ep1(:,3);
ep2 = ep2(:,3);

ep1 = ep1(1:2)/ep1(3);
ep2 = ep2(1:2)/ep2(3);

figure;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
subplot('Position', [0 .1 .5 1]);
imshow(image1);
hold on;
plot(ep1(1), ep1(2),'r+', 'MarkerSize', 50);

subplot('Position',[0.5 .1 .5 1]);
imshow(image2);
hold on;
plot(ep2(1), ep2(2),'r+', 'MarkerSize', 50);


% ep1 =
% 
%   480.4455
%   309.0720
% 
% 
% ep2 =
% 
%   831.8305
%   312.7878

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% part 3

features1_used = matchedfeatures1(points_used).Location;
features2_used = matchedfeatures2(points_used).Location;

figure;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
subplot('Position', [0 .1 .5 1]);
imshow(image1);
title('Inliers and Epipolar Lines in First Image');
hold on;
plot(ep1(1), ep1(2),'r+', 'MarkerSize', 50);
%plot(features1_used(:,1),features1_used.Location(:,2),'go')

epiLines2 = epipolarLine(fund_ransac,features1_used);
points_border = lineToBorderPoints(epiLines2,size(image1));
line(points_border(:,[1,3])',points_border(:,[2,4])');

subplot('Position', [0.5 .1 .5 1]);
imshow(image2);
title('Inliers and Epipolar Lines in Second Image');
hold on;
plot(ep2(1), ep2(2),'r+', 'MarkerSize', 50);
%plot(features2_used(:,1),features2_used(:,2),'go')

epiLines1 = epipolarLine(fund_ransac',features2_used);
points_border = lineToBorderPoints(epiLines1,size(image1));
line(points_border(:,[1,3])',points_border(:,[2,4])');

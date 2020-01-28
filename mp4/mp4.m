%% Loading Up Neural Net and Obtaining New Mesh 
mapping = ECE417_MP4_train ( av_train, av_validate, silenceModel, 20, 'neuralnet' ) ; 
results = ECE417_MP4_test ( testAudio, silenceModel, mapping ) ; 

%% Loading Up Original Mesh  
fileID = fopen('mesh.txt','r');
A = fscanf(fileID, '%f' ) ; 
fclose(fileID);

vertices = zeros(A(1), 2) ; 

for i = 2: 2: A(1) * 2 
    vertices( i/2, 1:2 ) = A(i:i+1) ; 
end 

tri_loc = A(1) * 2 + 2 ; 
num_triangle = A(tri_loc ) ; 
triangle = zeros(num_triangle, 3) ; 

for i = 3: 3 : num_triangle * 3 
    triangle(i/3, 1:3 ) = A(tri_loc + i - 2: tri_loc + i ) ; 
end 

% B = dlmread('mesh.txt') ; 
imshow(imread('mouth.jpg'));
hold on ; 

trimesh(triangle, vertices(:,1),vertices(:,2) );
scatter(vertices(:,1),vertices(:,2) );



%% Image Warping / Barycentric Coordinates 
% bilinear currently has New Facial Picture as input. 

% Order : Width, H1, H2 

new_coord = ones(3,3) ; 
old_coord = ones(3,3) ; 
new_vertices = zeros(33,2) ; 

V = VideoWriter('test3.avi' ) ; 
open(V) ; 

for i = 1 : 456
    [new_vertices(:,1), new_vertices(:,2)] = interpVert(vertices(:,1), vertices(:,2), 0, 0, 0, results(1,i), results(2,i), results(3,i), 1) ; 
         
    % Obtaining Barycentric Coordinates 
    [num_triangle] = bary_coord(new_vertices, triangle) ; 

    % Bilinear Interpolation
    output_image = bilinear(num_triangle, vertices, new_vertices, triangle) ; 
    
    % Saving Frame Image 

%     baseFileName = sprintf('test_%d.jpg', i-1);
%     fullFileName = fullfile('C:\Users\chris\Downloads\ECE_417\labs\mp4\mp4\video\', baseFileName);
%     imwrite(output_image, fullFileName);  
    
    writeVideo(V,output_image) ; 
       
end 

close(V) ; 



%% Extra Credit 
fileID = fopen('new_mesh.txt','r');
A = fscanf(fileID, '%f' ) ; 
fclose(fileID);

vertices = zeros(A(1), 2) ; 

for i = 2: 2: A(1) * 2 
    vertices( i/2, 1:2 ) = A(i:i+1) ; 
end 

tri_loc = A(1) * 2 + 2 ; 
num_triangle = A(tri_loc ) ; 
triangle = zeros(num_triangle, 3) ; 

for i = 3: 3 : num_triangle * 3 
    triangle(i/3, 1:3 ) = A(tri_loc + i - 2: tri_loc + i ) ; 
end 

imshow(imread('newface.jpg'));
hold on ; 

trimesh(triangle, vertices(:,1),vertices(:,2) );


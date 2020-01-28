function [ output_image ] = bilinear( num_triangle, vertices, new_vertices, triangle )
%BILINEAR Summary of this function goes here
%   Detailed explanation goes here

current = imread('newface.jpg') ;
current = rgb2gray(current) ; 
output_image = imread('newface.jpg') ;
output_image = rgb2gray(output_image) ; 
old_coord = ones(3,3) ; 
new_coord = ones(3,3) ; 
little_x = ones(3,1) ; 
% teeth = imread('teeth.jpg') ; 
% teeth = rgb2gray(teeth) ; 

for u = 1 : 130 
    for v = 1 : 86
        
        if ( num_triangle(v,u) ~= 0 ) 
            % Obtaining all needed Variables 
            tri_num = num_triangle(v,u) ; 
            
            new_coord(1:2,1) = new_vertices( triangle(tri_num,1),: ) ; 
            new_coord(1:2,2) = new_vertices( triangle(tri_num,2),: ) ; 
            new_coord(1:2,3) = new_vertices( triangle(tri_num,3),: ) ;
            
            old_coord(1:2,1) = vertices( triangle(tri_num,1),: ) ; 
            old_coord(1:2,2) = vertices( triangle(tri_num,2),: ) ; 
            old_coord(1:2,3) = vertices( triangle(tri_num,3),: ) ;
            
            little_x(1,1) = u ; 
            little_x(2,1) = v ; 
            
            % Calculating Lambda and new coordinates 
            lambda = inv(new_coord) * little_x + 0.001 ; 
            little_u = old_coord * lambda ; 
        
            % Bilinear Interpolation Calculation 
            current_u = floor(little_u(1,1) ) ;
            current_v = floor(little_u(2,1) ) ;
            
            f = little_u(1,1) - current_u ; 
            e = little_u(2,1) - current_v ; 
           
            if ( little_u(1,1) <= 130 && little_u(2,1) <= 86 ) 
                % First Part 
                first_part = (1-f) * (1-e) * current( current_v, current_u) ; 
            
                % Second Part 
                if current_v + 1 > 86 
                    second_part = (1-f) * e * current(current_v, current_u);
                else 
                    second_part = (1-f) * e * current(current_v+1, current_u) ; 
                end 
            
                % Third Part 
                if current_u + 1 > 130
                    third_part = f * (1-e) * current(current_v, current_u) ; 
                else 
                    third_part = f * (1-e) * current(current_v, current_u+1) ; 
                end         
            
                % Fourth Part 
                if (current_u+1 > 130) || (current_v+1 > 86) 
                    fourth_part = f * e * current(current_v, current_u)  ; 
                else 
                    fourth_part = f * e * current(current_v+1, current_u+1) ; 
                end
            
                output_image(v,u) = first_part + second_part + third_part + fourth_part ;
            end 
            
        
        else
            output_image(v,u) = 0 ; %teeth(v,u) ; 
        end     
        
    end 
end 


end


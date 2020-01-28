function [ num_triangle ] = bary_coord( new_vertices, triangle )
%BARY_COORD Summary of this function goes here
%   Detailed explanation goes here

% Barycentric Coordinates 
num_triangle = zeros(86, 130 ) ; 
little_x = ones(3,1) ; 
new_coord = ones(3,3) ; 

% Looping through all Output Pixels
for x = 1 : 130 
    for y = 1 : 86 
        
        % Attempting to find the Corresponding Triangle. 
        for tri = 1 : 42 
            new_coord(1:2,1) = new_vertices( triangle(tri,1),: ) ; 
            new_coord(1:2,2) = new_vertices( triangle(tri,2),: ) ; 
            new_coord(1:2,3) = new_vertices( triangle(tri,3),: ) ;
            
            little_x(1,1) = x ; 
            little_x(2,1) = y ; 
                
            temp_lambda = inv(new_coord) * little_x  + 0.001;  

            % Condition Matched. Store the Lambda values in Return Triangle 
            if ( 0 <= temp_lambda ) & (temp_lambda <= 1 + 0.001 )  
                num_triangle(y,x) = tri ; 
                break ; 
            end       
        end        

    end 
end 


return ; 
end


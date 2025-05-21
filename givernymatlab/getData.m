function result = getData(authToken, dataset, var_original, timepoint_original, temporal_method_original, spatial_method_original, ...
    spatial_operator_original, points, option)

    import matlab.net.http.*
    
    % Determine the number of points and convert the points array to a string
    numPoints = size(points, 1);
    points_str = join(arrayfun(@(i) sprintf('%.8f\t%.8f\t%.8f', points(i, :)), 1:numPoints, 'UniformOutput', false), newline);

    % Retrieve data through REST web service
    options = HTTPOptions('ConnectTimeout', 1000);
    request = RequestMessage('POST', [], points_str);
    
    functionname = 'GetVariable';
    
    if nargin == 8
        url = ['https://web.idies.jhu.edu/turbulence-svc/values?authToken=', authToken, '&dataset=', dataset,...
                '&function=', functionname, '&var=', var_original, ...
                '&t=', num2str(timepoint_original),  '&sint=', spatial_method_original, '&sop=', spatial_operator_original,...
                '&tint=', temporal_method_original];    
    elseif nargin == 9
          url = ['https://web.idies.jhu.edu/turbulence-svc/values?authToken=', authToken, '&dataset=', dataset,...
                '&function=', functionname, '&var=', var_original, ...
                '&t=', num2str(timepoint_original),  '&sint=', spatial_method_original, '&sop=', spatial_operator_original,...
                '&tint=', temporal_method_original, '&timepoint_end=', num2str(option(1)), '&delta_t=', num2str(option(2))];
    else
        error('Incorrect number of arguments.');
    end

    response = request.send(url, options);
    result = response.Body.Data;
    
    if response.StatusCode ~= matlab.net.http.StatusCode.OK
        if isfield(result, 'description')
            error(['HTTP Error ', char(response.StatusCode), ':', newline, ...
                   strjoin(result.description, newline)]);
        else
            error(['HTTP Error ', char(response.StatusCode), '.']);
        end
    end


    if nargin == 9
        times_plot = timepoint_original:option(2):option(1);
        result = reshapeAndPermute(result, var_original, spatial_operator_original, numPoints, length(times_plot));
    end
end

function result = reshapeAndPermute(data, var_original, spatial_operator_original, numPoints, numTimes)
    switch var_original
        case 'velocity'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [3, 9, 18, 3]);
        case 'vectorpotential'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [3, 9, 18, 3]);            
        case 'magneticfield'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [3, 9, 18, 3]);      
        case 'force'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [3, 9, 18, 3]);                
        case 'pressure'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'soiltemperature'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'sgsenergy'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'temperature'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'sgsviscosity'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);
        case 'density'
            result = reshapeByOperator(data, spatial_operator_original, numPoints, numTimes, [1, 3, 6, NaN]);           
        
        case 'position'
            if strcmp(spatial_operator_original, 'field')
                result = data;
            else
                handleErrorStruct(data, 'Invalid spatial operator for position query.');
            end
        otherwise
                handleErrorStruct(data, ['Unknown variable: ', var_original]);
       end
end

function result = reshapeByOperator(data, operator, numPoints, numTimes, dims)
    switch operator
        case 'field'
            result = reshape(data, [numPoints, numTimes, dims(1)]);
        case 'gradient'
            result = reshape(data, [numPoints, numTimes, dims(2)]);
        case 'hessian'
            result = reshape(data, [numPoints, numTimes, dims(3)]);
        case 'laplacian'
            if isnan(dims(4))
                error(['Laplacian not supported for this variable.']);
            end
            result = reshape(data, [numPoints, numTimes, dims(4)]);
        otherwise
            handleErrorStruct(data, ['Unknown spatial operator: ', operator]);
    end
    result = permute(result, [2, 1, 3]);  % [time, point, component]
end

function handleErrorStruct(data, msg)
    if isstruct(data) && isfield(data, 'description')
        error(['%s', newline, '%s'], msg, strjoin(data.description, newline));
    else
        error(msg);
    end
end

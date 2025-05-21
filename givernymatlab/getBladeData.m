function result = getBladeData(authToken, dataset, var, turbine_numbers, blade_numbers, original_times, blade_points)

    import matlab.net.http.*

    % construct payload as a MATLAB struct
    json_payload = struct( ...
        'auth_token', authToken, ...
        'dataset_title', dataset, ...
        'blade_variable', var, ...
        'turbines', turbine_numbers(:)', ...
        'blades', blade_numbers(:)', ...
        'blade_times', original_times(:)', ...
        'blade_actuator_points', blade_points(:)');


    % send POST request
    headers = matlab.net.http.HeaderField('Content-Type', 'application/json');
    request = RequestMessage('POST', headers, json_payload);
    options = HTTPOptions('ConnectTimeout', 1000);

    url = 'https://web.idies.jhu.edu/turbulence-svc/blade?include_metadata=0';
    response = request.send(url, options);

    % parse the JSON response
    raw_data = response.Body.Data;
    
    % --- print response and raw data for debugging ---
    %     disp('--- Raw HTTP Response ---');
    %     disp(response);
   
    %     disp('--- Parsed JSON Data ---');
    %     disp(raw_data);
    %     disp('Data preview:');
    %     disp(raw_data(1));  % show first entry
    
    
      % --- Error handling ---
    if response.StatusCode ~= matlab.net.http.StatusCode.OK
        if isfield(raw_data, 'description')
            error(['HTTP Error ', char(response.StatusCode), ': ', newline, ...
                   strjoin(raw_data.description, newline)]);
        else
            error(['HTTP Error ', char(response.StatusCode), '.']);
        end
    end

    % ----------------------------
    % convert raw numeric matrix to table
    % build column names
    actuator_var_names = arrayfun(@(p) sprintf('%s_%d', var, p), blade_points, 'UniformOutput', false);
    column_names = [{'time', 'turbine', 'blade'}, actuator_var_names];

    result_table = array2table(raw_data, 'VariableNames', column_names);

    % convert turbine and blade columns to integer
    result_table.turbine = int32(result_table.turbine);
    result_table.blade = int32(result_table.blade);

    % sort rows by turbine, blade, and time
    result_table = sortrows(result_table, {'turbine', 'blade', 'time'});

    % add index column as the first column
    num_times = numel(original_times);
    index_column = mod((0:height(result_table)-1)', num_times);

    % reorder columns: index, time, turbine, blade, then variables
    result = [table(index_column, 'VariableNames', {'index'}), result_table];
end

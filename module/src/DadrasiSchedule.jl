module DadrasiSchedule

# Import required packages
using Logging
using Random
using CSV
using DataFrames
using JuMP
using HiGHS
using LinearAlgebra
using Dates
using Base.Threads
using XLSX
using Statistics

# Disable logging

#Logging.disable_logging(Logging.AboveMaxLevel)
Logging.disable_logging(Logging.BelowMinLevel)

# Define the configuration structure
struct SchedulingConfig
    # Global parameters
    working_days::Int
    working_hour_per_Day::Int
    branch_run::Int
    LowerBound::Int  # stored as slot index from Start_times
    UpperBound::Int  # stored as slot index from Start_times
    
    # File paths
    main_data_path::String
    reps_data_path::String
    rooms_data_path::String
    schedule_data_path::String
    Betta::Int
    output_path::String 

    function SchedulingConfig(;
        working_days = 10,
        working_hour_per_Day = 120,  # 120 × 5 min = 10 hours
        branch_run = 12,
        
        LowerBound::Union{String,Time} = "08:00:00",
        UpperBound::Union{String,Time} = "18:00:00",
        main_data_path = "OBJ2.csv",
        reps_data_path = "Reps.csv",
        rooms_data_path = "Rooms.csv",
        schedule_data_path = "Barnameh_2.csv",
        Betta = 5,
        output_path::String = "./results.xlsx"
    )

        # Parse bounds if given as strings
        Start_times=Time(8, 0, 0)   # Start of working day
        lb_time = isa(LowerBound, String) ? Time(LowerBound, dateformat"HH:MM:SS") : LowerBound
        ub_time = isa(UpperBound, String) ? Time(UpperBound, dateformat"HH:MM:SS") : UpperBound

        # Calculate slot index relative to Start_times
        lb_slot = Int(Dates.value(lb_time - Start_times) ÷ (5 * 60 * 10^9))
        ub_slot = Int(Dates.value(ub_time - Start_times) ÷ (5 * 60 * 10^9))

        new(working_days, working_hour_per_Day, branch_run, lb_slot, ub_slot,
            main_data_path, reps_data_path, rooms_data_path, schedule_data_path, Betta, output_path)
    end
end


# Export public functions and types
export SchedulingConfig, Read_data, generate_model, main_script, excel_output, schedule_dadrasi, get_reorder_mapping

# Define module-level variables (const to avoid the error)
const DEFAULT_CONFIG = SchedulingConfig()

# Read_data function
function Read_data(config::SchedulingConfig=DEFAULT_CONFIG)
    @info "Starting Read_data function" config_type=typeof(config)
    
    # Extract parameters from config
    @debug "Extracting configuration parameters"
    working_days = config.working_days
    working_hour_per_Day = config.working_hour_per_Day
    branch_run = config.branch_run
    LowerBound = config.LowerBound
    UpperBound = config.UpperBound
    @info "Configuration loaded" working_days=working_days working_hour_per_Day=working_hour_per_Day branch_run=branch_run
    
    # Time intervals
    Del = 1:working_days*working_hour_per_Day
    @debug "Time intervals created" total_intervals=length(Del)
    
    # Load main data
    @info "Loading main data from CSV" path=config.main_data_path
    local data
    try
        data = CSV.File(config.main_data_path, header=true, types=Dict(
            :Column1 => Int32, :Column2 => Int32, :Column3 => String, :Column4 => String, :Column5 => Int32,
            :Column6 => Int32, :Column7 => Float64, :Column8 => String, :Column9 => Int32, :Column10 => String, 
            :Column11 => String, :Column12 => Int32, :Column13 => String )) |> DataFrame
        @info "Main data loaded successfully" rows=nrow(data) columns=ncol(data) 
    catch e
        @error "Failed to load main data" path=config.main_data_path error=e
        rethrow(e)
    end
    
    # filter cases with negative and zero NPW
    @info "Filtering cases with zero/negative NPW ..." length(data[:,1])
    sizedata = length(data[:,1])
    
    data = filter(row -> row[7] > 0 ,data)
    
    sizedata = sizedata - length(data[:,1])
    if sizedata >0
	@debug "Filtering data by branch" branch=branch_run    
    end 

	
	
    @debug "Filtering data by branch" branch=branch_run
    initial_rows = nrow(data)
    data = data[(data[:,12] .== branch_run), :]
    @info "Data filtered by branch" initial_rows=initial_rows filtered_rows=nrow(data) branch=branch_run
    
    # Add LAF_nums column
    @debug "Adding LAF_nums column..."
    data = hcat(data, DataFrame(LAF_nums=zeros(Int16, nrow(data))))
    @debug "LAF_nums column added"
    
    # Process data - update NPW and LAF counts
    @info "Starting NPW and LAF processing" total_rows=size(data,1)
    for i in 1:size(data,1)
        if i % 1000 == 0
            @debug "Processing row" current_row=i total_rows=size(data,1)
        end
        
        if data[i,1] != data[i,2] # update NPW
            row = findfirst(x -> x==data[i,2], data[:,1])
            if !isnothing(row)
                @debug "Updating parent NPW" row=i parent_row=row value_added=data[i,7]
                data[row,7] += data[i,7] # Update value of parent
            else # parent case is finished
                @debug "Parent case finished, self-referencing" row=i
                data[i,2] = data[i,1]
            end
        else # update number of cases in LAF
            num_laf = findall(x -> x==data[i,1], data[:,2])
            data[i,14] = length(num_laf)
            @debug "Updated LAF count" row=i laf_count=length(num_laf)
        end
    end
    @info "NPW and LAF processing completed"

    # Filter and sort data
    @debug "Starting data filtering and sorting"
    initial_size = nrow(data)
    data = filter(row -> row[1] == row[2], data)
    @debug "Filtered self-referencing rows" before=initial_size after=nrow(data)
    
    initial_size = nrow(data)
    data = filter(row -> !ismissing(row[6]), data)
    @debug "Filtered missing values in column 6" before=initial_size after=nrow(data)
    
    # Drop missing values
    initial_size = nrow(data)
    #data = dropmissing(data)
    @info "Dropped missing values" before=initial_size after=nrow(data)
    
    
    initial_size = nrow(data)
    data = data[1:min(5000, nrow(data)), :]    
    @info "Data truncated to top cases" original_size=initial_size final_size=nrow(data)
   
    
    data = sort(data, 5, rev=true)
    @info "Data sorted by column 5 (descending)"
    
    index_cases = copy(data[:,1])
    
    data = data[:,3:end]
    @info "Removed first two columns"
    
    # Convert Column3 to Int64
    @info "Converting Column3 to Int64"
    try
        data.Column3 = Int64.(round.(parse.(Float64, data.Column3)))
        @info "Column3 conversion successful"
    catch e
        @error "Failed to convert Column3 objection file to Float" error=e        
        rethrow(e)
        error("Failed to convert Column3 objection file to Float")
    end
    
    # Create C_Total and case information
    @debug "Creating case information structures"
    C_Total = 1:length(data[:,1])
    Case_Table = zeros(Float64, length(C_Total), 3)
    Case_Table[:,1] = C_Total # ID
    Case_Table[:,2] = data[:,1] # Real Name
    Case_Table[:,3] = data[:,5] # NPW
    @info "Case table created" total_cases=length(C_Total)    
    
    NPW = Case_Table[:,3]
    NLAF = data[:,12]
    @debug "NPW and NLAF arrays extracted"
    
    # Generate branch Info matrix
    @debug "Generating branch information matrix"
    branch = zeros(length(unique(data[:,4])), 2)
    branch[:,1] = 1:length(unique(data[:,4])) # ID branch
    branch[:,2] = unique(sort(data[:,4])) # Real Name Branch
    @info "Branch matrix created" total_branches=size(branch,1)
    
    # IC matrix
    @debug "Creating IC matrix..."
    IC = zeros(1, length(C_Total))
    for i in 1:length(C_Total)
        bra = findfirst(x -> x==data[i,4], branch[:,2])
        if !isnothing(bra)
            #IC[bra[1],i] = 1
            IC[1,i] = 1
        else
            @warn "Branch not found for case" case=i branch_value=data[i,4]
        end
    end
    @debug "IC matrix completed" dimensions=size(IC)    
    
    # Load schedule data
    @info "Loading schedule data" path=config.schedule_data_path
    local data_barnameh
    try
        #data_barnameh = CSV.File(config.schedule_data_path, header=true) |> DataFrame
        data_barnameh = CSV.File(config.schedule_data_path,
			    header=true,
			    types=[Int64, Int64, String, Int64, Time, Time]) |> DataFrame
        @info "Schedule data loaded" total_rows=nrow(data_barnameh)
        
    catch e
        @error "Failed to load schedule data" path=config.schedule_data_path error=e
        rethrow(e)
        error("Failed to load schedule data")
    end
    
    initial_schedule = nrow(data_barnameh)
    data_barnameh = data_barnameh[(data_barnameh[:,4] .== branch_run), :]
    @debug "Schedule filtered by branch" before=initial_schedule after=nrow(data_barnameh)
    
    data_barnameh = unique(data_barnameh, keep=:first)
    @info "Duplicate schedules removed" final_rows=nrow(data_barnameh)
    
    Reps_IDs = unique(data_barnameh[!,2])   
    
    # Load representatives data
    @info "Loading representatives data" path=config.reps_data_path
    local data_15
    try
        #data_15 = CSV.File(config.reps_data_path, header=true) |> DataFrame
        data_15 = CSV.File(config.reps_data_path,header=false, 
                           types=[Int64, Int64, String, Union{Missing, Int64}, 
			   Union{Missing, String}, 
			   Union{Missing, String}, 
			   Union{Missing, String}]) |> DataFrame

        @info "Representatives data loaded" total_rows=nrow(data_15)
    catch e
        @info "Failed to load representatives data" path=config.reps_data_path error=e
        rethrow(e)
        error("Failed to load representatives data")
    end
    
    initial_reps = nrow(data_15)
    #data_15 = data_15[(data_15[:,1] .== branch_run), :]
    data_15 = filter(row -> row[2] in Reps_IDs, data_15)
    @info "Representatives filtered by branch" before=initial_reps after=nrow(data_15) branch=branch_run
    
    num_reps = length(data_15[:,2])
    
    N = 1:num_reps # Representatives Set
    @info "Representatives set created" total_reps=num_reps

    reps = zeros(num_reps, 2)
    reps[:,1] = 1:num_reps # ID
    reps[:,2] = data_15[:,2] # Real ID
    
    # IN matrix
    @debug "Creating IN matrix"
    IN = zeros(Int, 1, num_reps)
    for i in 1:num_reps
        bra = findfirst(x -> x == data_15[i,1], branch[:,2])
        if !isnothing(bra)
            #IN[bra[1],i] = 1
            IN[1,i] = 1
        else
            @warn "Branch not found for representative" rep=i branch_value=data_15[i,1]
        end
    end
    @debug "IN matrix completed" dimensions=size(IN)
    
    # Process representative bands
    N1, N2, N3, N4 = Int[], Int[], Int[], Int[]

    band_errors = 0

    for i in 1:num_reps
        cell_val = string(data_15[i, 3])

        if ismissing(cell_val) || isempty(string(cell_val))
            @debug "Empty or missing band data at row $i"
            continue
        end

        #cleaned_data = replace(cell_val, r"^[\[\]\\\"]*|\"|\\|[\[\]]" => "")
	cleaned_data = cell_val
        band_strings = split(cleaned_data, "-")
        found_band = false 

        for bs in band_strings
            bs = strip(bs)  
            m = match(r"\d+", bs)  
            if isnothing(m)
                continue
            end

            band_num = parse(Int, m.match)
            band_transformed = band_num - 100  
    
            if band_transformed in (1, 2, 3)
                found_band = true
                if band_transformed == 1
                    push!(N1, i)
                elseif band_transformed == 2
                    push!(N2, i)
                elseif band_transformed == 3
                    push!(N3, i)
                end
            else band_transformed == 4
                @warn "Row $i has no relevant band (only found: $band_strings)"            
            end
        end
    end

    @info "Band processing completed" total_reps=num_reps band_errors=band_errors

    # Load rooms data
    @info "Loading rooms data" path=config.rooms_data_path
    local data_2
    try
        #data_2 = CSV.File(config.rooms_data_path, header=true) |> DataFrame
        data_2 = CSV.File(config.rooms_data_path, header=true,types=[Int64, Int64, String, String, String]) |> DataFrame
        @info "Rooms data loaded" total_rows=nrow(data_2)
    catch e
        @error "Failed to load rooms data" path=config.rooms_data_path error=e
        rethrow(e)
        error("Failed to load rooms data (Read Data)")
    end
    
    data_2 = data_2[(data_2[:,2] .== branch_run), :]
    
    num_rooms = length(data_2[:,1])
    num_biuld = length(unique(data_2[:,2]))
    @info "Rooms and buildings counted" rooms=num_rooms buildings=num_biuld
    
    R = 1:num_rooms # Set R
    S = 1:num_biuld # Set S
    
    Room = zeros(num_rooms, 3)
    Room[:,1] = 1:num_rooms # ID
    Room[:,2] = data_2[:,1] # Dadrasi
    Room[:,3] = data_2[:,2] # Building
    @debug "Room matrix created"
    
    # IR matrix
    @debug "Creating IR matrix"
    IR = zeros(1, num_rooms)
    ir_warnings = 0
    for i in 1:num_rooms
        bra = findfirst(x -> x==Room[i,2], branch[:,2])
        if !isnothing(bra)
            #IR[bra[1],i] = 1
            IR[1,i] = 1
        else
            ir_warnings += 1
        end
    end
    @info "IR matrix completed" dimensions=size(IR) warnings=ir_warnings
    
    # NR matrix - representative room assignments
    @debug "Processing NR matrix"
    for i in 7:size(data_15,2)
        data_15[!,i] = replace(data_15[!,i], "\\N" => missing)
    end
    @debug "Replaced \\N with missing values"
    
    NR = zeros(length(N), length(R))
    nr_assignments = 0
    for i in 1:length(N1)
    	if !ismissing(data_15[N1[i],7])
        	Room_addresses_str = replace(data_15[N1[i],7], r"^[\[\]\\\"]*|\"|\\|[\[\]]" => "")
        	room_codes = split(Room_addresses_str, "-")
        
        	for room_code in room_codes
            		# Trim whitespace from each room code
            		room_code = strip(room_code)
            		if isempty(room_code)
                		continue
            		end

            		Room_address = findfirst(x -> x == room_code, data_2[:,4])
            		if !isnothing(Room_address)
                		NR[N1[i], Room_address] = 1
                		nr_assignments += 1
            		else
                		@debug "Room address not found" rep=N1[i] room_code=room_code column=7
	        	end
        	end
    	end
    end

    @info "Total NR assignments made: $nr_assignments"
    
    # Buildings and IS/SR matrices
    @debug "Creating buildings and IS/SR matrices"
    dad = unique!(data_2[:,1])
    Buildings = zeros(Int32, num_biuld, 3)
    Buildings[:,1] = 1:num_biuld # ID
    Buildings[:,2] = unique(data_2[:,2]) # Real name
    @debug "Buildings matrix created" total_buildings=num_biuld

    IS = zeros(1, num_biuld)
    is_assignments = 0
    for i in dad
        mod_dad = findall(x -> x==i, data_2[:,1])
        biu = unique(data_2[mod_dad,2])
        
        for j in biu
            address = findfirst(x -> x==j, Buildings[:,2])
            mod = findfirst(x->x==i, branch[:,2])
            if !isnothing(address) && !isnothing(mod)
                #IS[mod,address] = 1
                IS[1,address] = 1
                is_assignments += 1
            end
        end
    end
    @info "IS matrix completed" dimensions=size(IS) assignments=is_assignments
    
    SR = zeros(num_biuld, num_rooms)
    for i in 1:num_rooms
        bra = findfirst(x -> x== data_2[i,2], Buildings[:,2])
        if !isnothing(bra)
            #SR[bra[1],i] = 1
            SR[1,i] = 1
        else
            @debug "Building not found for room" room=i building=data_2[i,2]
        end
    end
    @info "SR matrix completed" dimensions=size(SR)
    
    # Alleged matrix
    @info "Creating Alleged matrix"
    Alleged = zeros(length(C_Total), num_reps)
    alleged_assignments = 0
    for c in 1:length(C_Total)
        if ismissing(data[c,7]) || isnothing(data[c,7])
           continue
        end
         
        senf = data[c,7] # code e marja'       
	
        reps_c = findall(x -> !ismissing(x) && x == senf[1], data_15[:,4])
        
        if senf == 5 || senf == 6 # اتاق اصناف و مجامع حرفه‌ای
            #@info "for case $c, senf 5&6 detected
            inta_codes_case = data[c,8] # inta code case
            #if isequal(inta_codes_case,"\\N") || isequal(inta_codes_case,"\\") || ismissing(inta_codes_case)  || isnothing(inta_codes_case)
            if ismissing(inta_codes_case)
                for i in reps_c
                    Alleged[c,i] = 1
                    alleged_assignments += 1
                end
            else
                
                try
                    inta_codes_case = [parse(Int32,x) for x in split(inta_codes_case,"-")]
                    for i in reps_c
                        inta_codes_reps = data_15[i,6] # inta code representor
                       
                        #if isequal(inta_codes_reps,"\\N")  || isequal(inta_codes_reps,"\\") || ismissing(inta_codes_reps) || isnothing(inta_codes_reps) 
                        if ismissing(inta_codes_reps)
                            Alleged[c,i] = 1 
                            alleged_assignments += 1
                            continue
                        end

                        @info "for case $c, senf 5&6, inta code rep $i==> ok, split and search within inta codes $i"
                        inta_codes_reps = [parse(Int32,x) for x in split(inta_codes_reps,"-")]                        
                        if !isnothing(inta_codes_reps) && !isnothing(inta_codes_case)
                            for j in inta_codes_case
                                if !isnothing(findfirst(x -> x==j, inta_codes_reps))
                                    @info "for case $c, senf 5&6, inta code rep $i match"
                                    Alleged[c,i] = 1
                                    alleged_assignments += 1
                                    break
                                end
                            end
                        end
                    end
                catch e
                    @warn "Error parsing inta codes 5&6" case=c error=e
                    error("Error parsing inta codes 5&6")
                end
            end
        else
            @info "for case $c, senf non-5&6, allocate to all related reps"
            for i in reps_c
                Alleged[c,i] = 1
                alleged_assignments += 1
            end
        end
    end
    @info "Alleged matrix completed" dimensions=size(Alleged) assignments=alleged_assignments
    
    # Process presence data
    @debug "Processing presence data"
    num_elements = length(data_barnameh[:,1])
    Presence_2 = DataFrame(
        reps = zeros(Int16, num_elements),
        Build = zeros(Int16, num_elements),
        Day = zeros(Int16, num_elements),
        Ent = Vector{Time}(undef, num_elements),
        Exit = Vector{Time}(undef, num_elements)
    )
    @debug "Presence dataframe created" rows=num_elements
    
    # Days conversion
    @debug "Converting dates"
    try
        date_strings = data_barnameh[:,3]
        dates = [Date(d,"m/d/y") for d in date_strings]
        usd = sort(unique(dates))
        date_to_rank = Dict(date => rank for (rank,date) in enumerate(usd))
        ranked_dayes = [date_to_rank[d] for d in dates]
        Presence_2[:,3] = ranked_dayes
        @info "Date conversion completed" unique_dates=length(usd)
    catch e
        @error "Failed to convert dates" error=e
        rethrow(e)
        error("Failed to convert dates m/d/y")
    end
    
    # Building conversion
    @debug "Converting building references"
    building_not_found = 0
    for i in 1:num_elements
        addreth = findfirst(x -> x == data_barnameh[i,4], Buildings[:,2])
        if !isnothing(addreth)
            Presence_2[i,2] = addreth
        else
            building_not_found += 1
        end
    end
    @debug "Building conversion completed" not_found=building_not_found
    
    # Representative conversion
    @debug "Converting representative references"
    rep_not_found = 0
    for i in 1:num_elements
        addreth = findfirst(x -> x == data_barnameh[i,2], reps[:,2])
        if !isnothing(addreth)
            Presence_2[i,1] = addreth
        else
            rep_not_found += 1
        end
    end
    @debug "Representative conversion completed" not_found=rep_not_found
    
    # Hour conversion
    @debug "Converting time entries"
    function parse_time(time_str)
    
        
        if isa(time_str, Time)
            return time_str
        end
        if isa(time_str, String)
            time_str = replace(time_str, r":\d{2}\s" => " ")
            return Time(time_str, "H:MM p")
        end
        #time_str = replace(time_str, r":\d{2}\s" => " ")
        #return Time(time_str, "H:MM p")
    end
    
    entrance_times = data_barnameh[:,5]
    exit_times = data_barnameh[:,6]
    
    try        
        Presence_2[:,4] = parse_time.(entrance_times)
        Presence_2[:,5] = parse_time.(exit_times)
        @info "Time conversion completed"
    catch e
        @error "Failed to parse times" error=e
        rethrow(e)
        error("Failed to parse entrance/exit times")
    end
    
    # Filter dates
    @debug "Filtering presence dates"
    initial_presence = nrow(Presence_2)
    
    @info "Presence dates filtered" before=initial_presence after=nrow(Presence_2)
    
    # Create presence matrix
    @debug "Creating presence matrix"
    Start_times = Time(8,0,0)
    End_times = Time(18,0,0)
    time_slots = Start_times:Minute(5):End_times
    
    num_slots = working_days * length(time_slots)
    Presence_Matrix = zeros(Int, num_reps, num_biuld, num_slots)
    @info "Presence matrix initialized" dimensions=(num_reps, num_biuld, num_slots)
    
    # Fill Presence Matrix
    @debug "Filling presence matrix"
    presence_entries = 0
    skipped_entries = 0
    for (i,(ent,ex)) in enumerate(zip(Presence_2[:,4], Presence_2[:,5]))
        if Presence_2[i,3] <= working_days && Presence_2[i,1] != 0
            for (j,slot) in enumerate(time_slots)
                slot_end = slot + Minute(5)
                if ent <= slot_end && ex >= slot
                    if Presence_2[i,2] == 0 || Presence_2[i,1] == 0
                        skipped_entries += 1
                        continue
                    else
                        Presence_Matrix[Presence_2[i,1], Presence_2[i,2], (Presence_2[i,3]-1)*working_hour_per_Day + j] = 1
                        presence_entries += 1
                    end
                end
            end
        end
    end
    
    @info "Presence matrix filled" total_entries=presence_entries skipped=skipped_entries

    # Load timing data
    @info "Loading timing data" 

    Betta = Int16(config.Betta/ 5)
    @info "timing data Betta loaded"
    
    # Forbidden matrix
    @info "Creating forbidden matrix"
    forbidden = zeros(length(C_Total), num_reps)
    forbidden_count = 0
    for i in 1:length(C_Total)
        if ismissing(data[i,9])
            continue
        end
        tes = match(r"\d+", data[i,9])
        if !isnothing(tes) && length(string(data[i,9])) > 8
            tes = parse(Int, tes.match)
            if !isempty(tes)
                try
                    commenters = [parse(Int64,x) for x in split(data[i,9],"-") if !isempty(x)]
                    for j in commenters
                        addreth = findfirst(x -> x==j, reps[:,2])
                        if !isnothing(addreth)
                            forbidden[i,addreth] = 1
                            forbidden_count += 1
                        end
                    end
                catch e
                    @warn "Error parsing forbidden data" case=i value=data[i,9] error=e
                    error("Error parsing forbidden agents (black list)")
                end
            end
        end
    end
    @info "Forbidden matrix completed" dimensions=size(forbidden) total_forbidden=forbidden_count
    
    # CS matrix
    @debug "Creating CS matrix"
    CS = zeros(length(C_Total), num_biuld)
    cs_assignments = 0
    for i in 1:length(C_Total)
        addreth = findfirst(x -> x== data[i,10], Buildings[:,2])
        if !isnothing(addreth)
            CS[i,addreth] = 1
            cs_assignments += 1
        end
    end
    @info "CS matrix completed" dimensions=size(CS) assignments=cs_assignments
    
    # Create T_Delta matrix
    @debug "Creating T_Delta matrix"
    T_total = 1:working_days
    T_Delta = zeros(Int32, length(T_total), working_days*length(Del))
    for t in T_total
        T_Delta[t,(t-1)*working_hour_per_Day+1:t*working_hour_per_Day] .= 1
    end
    @info "T_Delta matrix completed" dimensions=size(T_Delta)
    
    # Other matrices
    @info "Creating Prefered agents matrices-Gamma[c,n]"
    Gamma = zeros(length(C_Total), num_reps)
    for i in 1:length(C_Total)
        if !isnothing(data[i,11])
            agents = parse.(Int, split(data[i,11],"-"))
            if !isempty(agents) && length(string(data[i,11])) > 8
                try
                    for j in agents
                        addreth = findfirst(x -> x==j, reps[:,2])
                        if !isnothing(addreth)
                            Gamma[i,addreth] = 1
                        end
                    end            
                catch e
                    @warn "Error parsing Gamma-PreferedAgents data" case=i value=a error=e
                    error("Error parsing Gamma-Prefered Agents (white list)")
                end
            end
        end    
    end
    @info "Gamma Craeted Successfuly" gamma_size=size(Gamma)
    
    @info "Creating redundant matrix Alpha"
    Alpha = zeros(10000, length(C_Total))
    for lm in 1:length(C_Total)
        Alpha[rand(1:10000),lm] = 1
    end
    Theta = 2*ones(length(C_Total), 1)
    @info "Auxiliary matrices created" alpha_size=size(Alpha) theta_size=size(Theta)
    
    # Return structured data with all necessary fields
    @info "Preparing final data structure"
    Data = (
        # Sets
        I = 1,
        R = R,
        N = N,
        P = 1:10000,
        Del = Del,
        G = ["out", "usu"],
        S = S,
        SR = SR,
        IS = IS,
        T_total = T_total,
        C_toal = C_Total,
        
        # Scalars
        BigM = 1e5,
        
        # Parameters
        T_Delta = T_Delta,
        IN = IN,
        IR = IR,
        IC = IC,
        working_hour_per_Day = working_hour_per_Day,
        Alpha = Alpha,
        N1 = N1,
        N2 = N2,
        N3 = N3,
        Gamma = Gamma,
        Pre_Sch = Presence_Matrix,
        Theta = Theta,
        Betta = Betta,
        forbidden = forbidden,
        Alleged = Alleged,
        NPW = NPW,
        NLAF = NLAF,
        NR = NR,
        CS = CS,
        Index_Cases = index_cases,
        
        # Additional fields for compatibility
        num_rooms = num_rooms,
        working_days = working_days,
        branch_run = branch_run,
        LowerBound = LowerBound,
        UpperBound = UpperBound,
        
        #imported data
        case_data = data,
        data_15 = data_15,
        data_barnameh = data_barnameh,
        data_2 = data_2
    )
    
    case_data=copy(data)
    @info "Data structure created successfully"
    
    @info "Read_data function completed successfully" total_cases=length(C_Total) total_reps=num_reps total_rooms=num_rooms total_buildings=num_biuld
    #return Data , case_data, data_15
    return Data 
end


function generate_model(T, C, Del, I, IC, Data, X_bar, Y_bar, XY_bar, Previous_Cases)
 @info "Starting generate_model function" timestamp=now() T=T C_size=length(C) Del_size=length(Del) I=I
 
 Betta = Data.Betta
 @debug "Extracted Betta from Data" Betta_size=size(Betta)
 
 start_time = now()
 @info "Model generation start time recorded" start_time=start_time
 
 # Unpack data structures
 @info "Unpacking data structures"
 P = Data.P
 G = Data.G
 T_total = Data.T_total
 BigM = Data.BigM
 IN = Data.IN
 IR = Data.IR
 Alpha = Data.Alpha
 N1 = Data.N1
 N2 = Data.N2
 N3 = Data.N3
 Gamma = Data.Gamma
 Pre_Sch = Data.Pre_Sch
 forbidden = Data.forbidden
 Alleged = Data.Alleged
 NPW = Data.NPW
 NLAF = Data.NLAF
 NR = Data.NR
 
 @info "Data structures unpacked successfully" BigM=BigM P_size=length(P) G_size=length(G) T_total_size=length(T_total)
 @debug "Matrix dimensions" IN_size=size(IN) IR_size=size(IR) Alpha_size=size(Alpha) Gamma_size=size(Gamma)
 @debug "Band sizes" N1_size=length(N1) N2_size=length(N2) N3_size=length(N3)
 
 # Filter rooms for current branch
 @info "Filtering rooms for current branch" branch_I=I
 R1 = vec(findall(x -> x == 1, skipmissing(IR[I, :] |> collect)))
 R = 1:length(R1)
 @info "Rooms filtered" R1_size=length(R1) R_size=length(R)
 

 # Filter representatives for current branch
 @info "Filtering representatives for current branch"
 N = vec(findall(x -> x == 1, IN[I, :] |> collect))
 @info "Representatives filtered" N_size=length(N)
 
 # Filter representative bands
 @info "Filtering representative bands for current branch"
 N1 = vec(filter(x -> x in N, N1))
 N2 = vec(filter(x -> x in N, N2))
 N3 = vec(filter(x -> x in N, N3))
 

 @info "Representative bands filtered" filtered_N1=length(N1) filtered_N2=length(N2) filtered_N3=length(N3)
 
 N01 = N
 N11 = collect(skipmissing(N1))
 N21 = collect(skipmissing(N2))
 N31 = collect(skipmissing(N3))
 @debug "Representative sets created" N01_size=length(N01) N11_size=length(N11) N21_size=length(N21) N31_size=length(N31)
 
 # Define optimization model
 @info "Creating optimization model with HiGHS solver"
 model = Model(HiGHS.Optimizer)
 @debug "HiGHS optimizer assigned"
 
 @info "Setting solver attributes"
 set_optimizer_attribute(model, "mip_rel_gap", 0.05) # relative gap (OPTCR)
 set_optimizer_attribute(model, "mip_abs_gap", 0.05) # Absolute gap tolerance (OPRCA)
 set_optimizer_attribute(model, "output_flag", false) # Turn off solver output
 set_optimizer_attribute(model, "parallel", "on")
 set_optimizer_attribute(model, "threads", 0)
 @info "Solver attributes configured" mip_rel_gap=0.05 mip_abs_gap=0.05 parallel="on"
 
 # Binary Variables
 @info "Creating binary variables"
 @variable(model, X[c in C, r in R, del in Del], Bin) # Binary variable X(c,r,del)
 @debug "X variable created" X_dimensions=(length(C), length(R), length(Del))
 
 @variable(model, Y[n in N, r in R, del in Del], Bin) # Binary variable Y(n,r,del)
 @debug "Y variable created" Y_dimensions=(length(N), length(R), length(Del))
 
 @variable(model, Z[n in N, r in R, del in Del], Bin) # Binary variable Z(n,r,del)
 @debug "Z variable created" Z_dimensions=(length(N), length(R), length(Del))
 
 @variable(model, Z_escape[n in N, r in R, del in Del], Bin) # Binary variable Z(n,r,del)
 @debug "Z_escape variable created" Z_escape_dimensions=(length(N), length(R), length(Del))
 
 # Continuous Variables
 @info "Creating continuous variables"
 @variable(model, XY[n in N, del in Del] >= 0) # Representative workload
 @debug "XY variable created" XY_dimensions=(length(N), length(Del))
 
 @variable(model, Z_bar[r in R, del in Del] >= 0) # Room utilization penalty
 @debug "Z_bar variable created" Z_bar_dimensions=(length(R), length(Del))
 
 @variable(model, escape[c in C, n in N], Bin) # Binary variable escape(c,n)
 @debug "escape variable created" escape_dimensions=length(C)
 
 @variable(model, OF) # Objective function variable
 @debug "OF variable created"
 
 end_time = now()
 @info "Variables created successfully" creation_time=end_time-start_time
 
 # Scale NPW values for objective function
 @info "Scaling NPW values for objective function"
 max_val = maximum(NPW[:, 1])
 min_val = minimum(NPW[:, 1])
 @debug "NPW statistics" max_val=max_val min_val=min_val range=max_val-min_val
 
 scaled_NPW = [5*(1-((max_val - x) / (max_val - min_val))) for x in NPW[:, 1]]
 scaled_NPW2 = [5*((max_val - x) / (max_val - min_val)) for x in NPW[:, 1]]
 @debug "NPW scaling completed" scaled_NPW_size=length(scaled_NPW) scaled_NPW2_size=length(scaled_NPW2)
 
 AVg = mean(scaled_NPW)
 @info "NPW scaling statistics" average=AVg max_scaled=maximum(scaled_NPW) min_scaled=minimum(scaled_NPW)
 
 # Objective Function
 @info "Setting objective function"
 @objective(model, Max, OF)
 
 @constraint(model, OF == sum(X[c, r, del] * scaled_NPW[c] for c in C, r in R, del in Del) 

 - sum( Z_escape[n,r,del]*(AVg/3) for n in N, r in R, del in Del)
 - sum((0.25*scaled_NPW[c])*escape[c,n] for c in C, n in N1) - sum((0.05*scaled_NPW[c])*escape[c,n] for c in C, n in N2) - sum((0.15*scaled_NPW[c])*escape[c,n] for c in C, n in N3)
 )
 
 @info "Objective function constraint added"
 # - sum( Z_escape[n,r,del]*(AVg/5) for n in N, r in R, del in Del) -  sum(2*exp(-scaled_NPW[c]/2)*escape[c] for c in C)
  
 #@constraint(model, conTEST[i in I], sum(Y[6,r,del] for r in R, del in Del) >= 1)
  
 # Constraint 1: Room capacity - at most one case per room per time slot
 @info "Adding constraint 1: Room capacity"
 @constraint(model, con1[i in I, r in R, del in Del; Bool(IR[i,R1[r]])], 
 sum(X[c,r,del] for c in C) <= 1)
 @debug "Constraint 1 added - room capacity"
 
 # Constraint 2: Case processing time conflicts
 @info "Adding constraint 2: Case processing time conflicts"
 Delp = (T-1)*Data.working_hour_per_Day+1 : Data.working_hour_per_Day*T
 @debug "Delp range created" Delp_range=Delp
 
 @constraint(model, con2[i in I, r in R, del in Del, delp in Delp; 
 (del >=2) && (delp <= (del -1)) && (delp >= max(1,del - Betta[i,1] +1))], 
 sum(X[cp,r,del] for cp in C) <= BigM*(1 - sum(X_bar[c,r,delp] for c in Data.C_toal)))
 @debug "Constraint 2 added - processing time conflicts"
 
 # Constraint 3: Each case scheduled at most once
 @info "Adding constraint 3: Each case scheduled at most once"
 @constraint(model, con3[c in C], sum(X[c, r, del] for r in R, del in Del) <= 1)
 @debug "Constraint 3 added - unique case scheduling"
 
 # Constraint 4: Representative availability in buildings
 @info "Adding constraint 4: Representative availability in buildings"
 @constraint(model, con4[i in I, n in N, del in Del, s in Data.S; Bool(Data.IS[i,s])], 
 sum(Y[n, r, del] for r in R if Bool(IR[i, R1[r]]) && Bool(Data.SR[s,R1[r]])) <= Pre_Sch[n, s, del])
 @debug "Constraint 4 added - representative availability"
 
 # Constraint 6_1: Representative requirements per case (3 representatives needed)
 @info "Adding constraint 6_1: Representative requirements per case"
 @constraint(model, con6_1[i in I, c in C, del in Del, r in R; Bool(IR[i,R1[r]])], 
 3 * X[c,r,del] <= sum(Y[n1,r,del] for n1 in N1) + sum(Y[n2,r,del] for n2 in N2) + 
 sum(Y[n3,r,del] for n3 in N3 if Bool(Alleged[c, n3])))
 @debug "Constraint 6_1 added - 3 representatives required"
 
 # Constraints 7-9: Representative band requirements and limits
 @info "Adding constraints 7-9: Representative band requirements"
 @constraint(model, con7[i in I,del in Del,r in R; Bool(IR[i,R1[r]])], 
 sum(X[c,r,del] for c in C)/10 <= sum(Y[n1,r,del] for n1 in N1))
 @constraint(model, con7_1[i in I,del in Del,r in R; Bool(IR[i,R1[r]])], 
 sum(Y[n1,r,del] for n1 in N1) <= 1)
 @debug "Constraints 7 and 7_1 added - Band 1 requirements"
 
 @constraint(model, con8[i in I,del in Del,r in R; Bool(IR[i,R1[r]])], 
 sum(X[c,r,del] for c in C)/10 <= sum(Y[n2,r,del] for n2 in N2))
 @constraint(model, con8_1[i in I,del in Del,r in R; Bool(IR[i,R1[r]])], 
 sum(Y[n2,r,del] for n2 in N2) <= 1)
 @debug "Constraints 8 and 8_1 added - Band 2 requirements"
 
 @constraint(model, con9[i in I,del in Del,r in R; Bool(IR[i,R1[r]])], 
 sum(X[c,r,del] for c in C)/10 <= sum(Y[n3,r,del] for n3 in N3))
 @constraint(model, con9_1[i in I,del in Del,r in R; Bool(IR[i,R1[r]])], 
 sum(Y[n3,r,del] for n3 in N3) <= 1)
 @debug "Constraints 9 and 9_1 added - Band 3 requirements"
 
 # Constraints 13-15: Forbidden assignments and eligibility
 @info "Adding constraints 13-15: Forbidden assignments and eligibility"
 @constraint(model, con13[i in I, c in C, del in Del, r in R; Bool(IR[i,R1[r]])], 
 X[c,r,del] <= sum(Y[n3,r,del] for n3 in N3 if Bool(Alleged[c, n3]) && !Bool(forbidden[c,n3])))
 @debug "Constraint 13 added - Band 3 forbidden assignments"
 
 @constraint(model, con14[i in I, c in C, del in Del, r in R; Bool(IR[i,R1[r]])], 
 X[c,r,del] <= sum(Y[n2,r,del] for n2 in N2 if !Bool(forbidden[c,n2])))
 @debug "Constraint 14 added - Band 2 forbidden assignments"
 
 @constraint(model, con15[i in I, c in C, del in Del, r in R; Bool(IR[i,R1[r]])], 
 X[c,r,del] <= sum(Y[n1,r,del] for n1 in N1 if !Bool(forbidden[c,n1])))
 @debug "Constraint 15 added - Band 1 forbidden assignments"
 
 # Time constraints - ensure case completion within same day
 @info "Adding time constraints: case completion within same day"
 @constraint(model, Con_New2[i in I, c in C, r in R, del in Del; Bool(IR[i, R1[r]])], 
 sum(t for t in T if Bool(Data.T_Delta[t, del + Betta[i,1] - 1])) >= 
 sum(t for t in T if Bool(Data.T_Delta[t, del])) - BigM * (1 - X[c, r, del]))
 @debug "Constraint Con_New2 added - time completion lower bound"
 
 @constraint(model, Con_New3[i in I, c in C, r in R, del in Del; Bool(IR[i, R1[r]])], 
 sum(t for t in T if Bool(Data.T_Delta[t, del + Betta[i,1] - 1])) <= 
 sum(t for t in T if Bool(Data.T_Delta[t, del])) + BigM * (1 - X[c, r, del]))
 @debug "Constraint Con_New3 added - time completion upper bound"
 
 # Constraint 31: Representative availability with escape option
 @info "Adding constraint 31: Representative availability with escape option"
 ##@constraint(model, con31[del in Del, c in C, r in R, i in I; Bool(IR[i,R1[r]])], 
 ##sum(Y[n,r,del] for n in N if Bool(Gamma[c,n])) >= 3*X[c,r,del] - BigM * escape[c])
 @constraint(model, con31[del in Del, c in C, r in R, i in I, n in N; Bool(IR[i,R1[r]]) && Bool(Gamma[c,n]) ], 
 Y[n,r,del] >= X[c,r,del] - BigM * escape[c,n])
 @debug "Constraint 31 added - representative availability with escape"
 
 @constraint(model, con31_1[del in Del, c in C, r in R, i in I; Bool(IR[i,R1[r]]) && Bool(IC[i,c])], 
 sum(Y[n,r,del] for n in N if Bool(IN[i,n])) >= 3*X[c,r,del])
 @debug "Constraint 31_1 added - branch-specific representative availability"
 
 # Constraint 30: Representative scheduling conflicts
 @info "Adding constraint 30: Representative scheduling conflicts"
 @constraint(model, con30[i in I, del in Del, n in N], 
 sum(Y[n,r,del] for r in R if Bool(IR[i,R1[r]])) + 
 sum(Y_bar[n,r,delp] for r in R, delp in Delp if 
 (delp <= del -1) && (delp >= del - Betta[i,1] + 1) && (Betta[i,1]>1)) <= 1)
 @debug "Constraint 30 added - representative scheduling conflicts"
 
 # Additional constraint: Representative-case ratio
 @info "Adding constraint: Representative-case ratio"
 @constraint(model, Con_New1[i in I,del in Del,r in R; Bool(IR[i,R1[r]])], 
 sum(Y[n,r,del] for n in N) <= 3 * sum(X[c,r,del] for c in C))
 @debug "Constraint Con_New1 added - representative-case ratio"
 
 # Rest period constraints for representatives
 @info "Adding rest period constraints for representatives"
 @constraint(model, con17[i in I, n in N, r in R, del in Del; 
 Bool(del-((T-1)*Data.working_hour_per_Day) >= 12*Betta[i,1])], 
 12 - sum(Y_bar[n,r,delp] for delp in Del if delp>=del-12*Betta[i,1] && delp<=del) <= 
 (1-Z[n,r,del])*BigM)
 @debug "Constraint 17 added - rest period constraint 1"
 
 @constraint(model, con18[i in I, n in N, r in R, del in Del; 
 Bool(del-((T-1)*Data.working_hour_per_Day) >= 12*Betta[i,1])], 
 Y_bar[n,r,del-Betta[i,1]] <= Y[n,r,del] + (Z[n,r,del]+ Z_escape[n,r,del])*BigM)
 @debug "Constraint 18 added - rest period constraint 2"
 
 @constraint(model, con19[i in I, n in N, r in R, del in Del; 
 Bool(del-((T-1)*Data.working_hour_per_Day) < 12*Betta[i,1]) && 
 Bool(del-((T-1)*Data.working_hour_per_Day) > Betta[i,1])], 
 Y_bar[n,r,del-Betta[i,1]] <= Y[n,r,del] + Z_escape[n,r,del]*BigM)
 @debug "Constraint 19 added - rest period constraint 3"
 
 # Unique assignment constraints
 @info "Adding unique assignment constraints"
 @constraint(model, Con_Jadid[i in I,del in Del,n in N], 
 sum(Y[n,r,del] for r in R) <= 1)
 @debug "Constraint Con_Jadid added - unique representative assignment"
 
 @constraint(model, Con_Jadid_2[r in R, i in I,del in Del,c in C], 
 sum(X[c,r,del] for del in Del) <= sum(Data.CS[c,s] for s in Data.S if Bool(Data.SR[s,R1[r]])))
 @debug "Constraint Con_Jadid_2 added - case-building eligibility"
 
 # Workload tracking constraints for Band 1 representatives
 @info "Adding workload tracking constraints for Band 1 representatives"
 @constraint(model, con32[i in I, del in Del, n1 in N1], 
 XY[n1,del] <= BigM*(sum(Y[n1,r,del] for r in R)))
 @debug "Constraint 32 added - workload upper bound"
 
 @constraint(model, con33[i in I, del in Del, n1 in N1, r in R; Bool(IR[i,R1[r]])], 
 XY[n1,del] <= sum(NLAF[c]*X[c,r,del] for c in C) + BigM*(1-Y[n1,r,del]))
 @debug "Constraint 33 added - workload calculation upper bound"
 
 @constraint(model, con34[i in I, del in Del, n1 in N1, r in R; Bool(IR[i,R1[r]])], 
 XY[n1,del] >= sum(NLAF[c]*X[c,r,del] for c in C) - BigM*(1-Y[n1,r,del]))
 @debug "Constraint 34 added - workload calculation lower bound"
 
 @constraint(model, con35[i in I, del in Del, n1 in N1], sum( (XY_bar[n1,delp]) for delp in Delp if delp < del) + XY[n1,del] <= 20)
 @debug "Constraint 35 added - cumulative workload limit"
 
 # Room eligibility constraints for Band 1 representatives
 @info "Adding room eligibility constraints for Band 1 representatives"
 @constraint(model, con36[i in I, n1 in N1], 
 sum(Y[n1,r,del] for del in Del, r in R if Bool(IR[i,R1[r]]) && !Bool(NR[n1,R1[r]])) <= 0)
 @debug "Constraint 36 added - room eligibility for Band 1"
 
 # Room utilization penalty
 @info "Adding room utilization penalty constraint"
 @constraint(model, con37[i in I, n in N, r in R, del in Del; Bool(IR[i,R1[r]])], 
 Z_bar[r,del] >= (1/2)*sum(scaled_NPW[c]*X[c,r,del] for c in C) - BigM*(1- Z_escape[n,r,del]))
 @debug "Constraint 37 added - room utilization penalty"
 
 end_time = now()
 @info "Model generation completed successfully" total_time=end_time-start_time constraints_added=37 variables_created=6
 
 return model, X, Y, XY
end



function main_script(Data)
    @info "Starting main_script function" timestamp=now()
    
    @info "Extracting configuration parameters"
    working_days = Data.working_days
    working_hour_per_Day = Data.working_hour_per_Day
    @info "Working schedule parameters" working_days=working_days working_hour_per_Day=working_hour_per_Day
    
    T_total = Data.T_total
    C_total = Data.C_toal
    Del = Data.Del
    T_Delta = Data.T_Delta
    @info "Time and case parameters" T_total=length(T_total) C_total=length(C_total) Del=length(Del)
    
    @info "Initializing tracking variables"
    Assigned_cases = []
    N_Assigned = 0
    ccc = 0
    @debug "Tracking variables initialized" Assigned_cases=length(Assigned_cases) N_Assigned=N_Assigned
    
    num_rooms = length(Data.R)
    @info "Room configuration" num_rooms=num_rooms
    
    @info "Initializing solution matrices"
    X_bar2 = zeros(length(Data.C_toal), num_rooms, length(Data.Del))
    Y_bar2 = zeros(length(Data.N), num_rooms, length(Data.Del))
    XY_bar2 = zeros(length(Data.N), length(Data.Del))
    @info "Solution matrices initialized" X_bar2_size=size(X_bar2) Y_bar2_size=size(Y_bar2) XY_bar2_size=size(XY_bar2)
    
    @info "Starting daily scheduling loop" total_days=working_days
    for T in 1:working_days
        @info "Processing day" current_day=T total_days=working_days
        
        Del = (T-1)*working_hour_per_Day+1 : working_hour_per_Day*T
        @debug "Time slots for current day" Del_range=Del Del_length=length(Del)
        
        @info "Creating case selection matrix"
        selected_case_ids = zeros(Int32, 1, length(C_total))
        for i in 1:length(C_total)
            selected_case_ids[i] = Int32(i)
        end
        @debug "Case IDs selected" total_cases=length(selected_case_ids)
        
        IC_new = Data.IC
        @debug "IC matrix assigned" IC_size=size(IC_new)
        
        @info "Building case-branch mapping matrix"
        mm = zeros(length(selected_case_ids), 3)
        mm[:, 1] = selected_case_ids
        for i in 1:length(selected_case_ids)
            mm[i, 2] = findfirst(x -> x == 1, Data.IC[:, selected_case_ids[i]])
            mm[i, 3] = Data.NPW[selected_case_ids[i], 1]
        end
        @debug "Case mapping matrix created" mm_size=size(mm)
        
        @info "Creating case dataframe and grouping by branch"
        
        ass = zeros(length(Del))
        @debug "Assignment tracking array initialized" ass_length=length(ass)
        
        @info "Processing branch I=1"
        for I in 1:1
            @debug "Current branch" branch_I=I
            
            #global Assigned_cases,N_Assigned
            #global X_bar2, Y_bar2, XY_bar2
            
            @info "Finding rooms for current branch"
            R1 = vec(findall(x -> x == 1, skipmissing(Data.IR[I, :] |> collect)))
            @info "Rooms filtered for branch" branch=I num_rooms_branch=length(R1)
            
            @info "Initializing branch-specific solution matrices"
            X_bar = zeros(length(Data.C_toal), length(R1), length(Data.Del))
            Y_bar = zeros(length(Data.N), length(R1), length(Data.Del))
            XY_bar = zeros(length(Data.N), length(Data.Del))
            @debug "Branch matrices initialized" X_bar_size=size(X_bar) Y_bar_size=size(Y_bar) XY_bar_size=size(XY_bar)                       
            
            @info "Starting time slot optimization loop" Del_range=Del
            for del in Del
                @debug "Processing time slot" current_del=del day=T
                
                if (del - (T-1)*working_hour_per_Day) < Data.LowerBound + 1 ||
                   (del - (T-1)*working_hour_per_Day) > Data.UpperBound + 1
                    @debug "Time slot outside bounds, skipping" del=del LowerBound=Data.LowerBound UpperBound=Data.UpperBound

                    continue
                end
                
                @info "Finding eligible cases for current time slot" del=del
                C_New_del = []
                Previous_Cases = vec(filter(x -> Data.IC[I, x] == 1, Assigned_cases))
                @debug "Previous cases filtered" num_previous=length(Previous_Cases)

                @info "Finding buildings for current branch"
                sakhtmons = findall(x -> x == 1, Data.IS[I, :])
                @debug "Buildings found for branch" branch=I num_buildings=length(sakhtmons)
                              
                num_case =round( (Data.UpperBound-Data.LowerBound)/(Data.Betta))
		for ss in sakhtmons
		    
		    num_otagh = length(findall(x -> x == 1, Data.SR[ss, :]))

		    marja_list = unique!(Data.data_15[:,4]) # unique marja codes of reps
		    marja_list = marja_list[.!ismissing.(marja_list)] # drop "missing" itself

		    emp = findall(x -> x == 1, Data.CS[:, ss])
		    emp = setdiff(emp, Previous_Cases)
		    
		    tmp = Data.case_data[emp,:]

		    for i in marja_list

			nominated_cases = filter(row -> row[7] == i, tmp) #filter marja
			
			nominated_cases = sort(nominated_cases, 5, rev=true) #NPW sort
			nominated_cases_IDs = Set(nominated_cases[!, 1]) #nominated case IDs
			
			Case_IDs = Data.case_data[!, 1] .∈ [nominated_cases_IDs]
			Case_IDs = findall(Case_IDs)			 

			C_New_del = vcat(C_New_del,Case_IDs[1:min(Int(4*num_otagh*num_case),length(Case_IDs))])

		    end
		end

		#C_New_del = sort(C_New_del, by=c -> Data.NPW[c, 1], rev=true)
		#C_New_del = C_New_del[1:min(20*num_otagh*num_case, length(C_New_del))]

                if isempty(C_New_del)
                    @debug "No eligible cases found, continuing to next time slot"
                    continue
                end
                
                @info "Generating optimization model" cases=length(C_New_del) time_slot=del
                model, X, Y, XY = generate_model(T, C_New_del, del, I, IC_new, Data, X_bar, Y_bar, XY_bar, Previous_Cases)
                @debug "Model generated successfully"
                
                @info "Optimizing model"
                optimize!(model)
                @info "Optimization completed" status=termination_status(model)
                
                if termination_status(model) != MOI.OPTIMAL
                    @warn "Model did not reach optimal solution" status=termination_status(model) time_slot=del
                    continue
                end
                
                @info "Processing X variable solutions"
                assigned_in_slot = 0
                for k in C_New_del
                    if round(sum(value.(X[k,:,del]))) == 1
                        @debug "Case assigned" case=k time_slot=del
                        assigned_in_slot += 1
                        for r in 1:length(R1)
                            X_bar[k, r, del] = value.(X[k, r, del])
                            X_bar2[k, R1[r], del] = value.(X[k, r, del])
                        end
                    end
                end
                @info "X solutions processed" cases_assigned_in_slot=assigned_in_slot
                
                @info "Processing Y variable solutions (representatives)"
                repss = vec(findall(x -> x == 1, skipmissing(Data.IN[I,:] |> collect)))
                @debug "Representatives found for branch" branch=I num_reps=length(repss)
                
                assigned_reps = 0
                for n in repss
                    if round(sum(value.(Y[n,:,del]))) == 1
                        @debug "Representative assigned" rep=n time_slot=del
                        assigned_reps += 1
                        for r in 1:length(R1)
                            Y_bar[n, r, del] = value.(Y[n, r, del])
                            Y_bar2[n, R1[r], del] = value.(Y[n, r, del])
                        end
                    end
                end
                @info "Y solutions processed" reps_assigned_in_slot=assigned_reps
                
                @info "Processing XY variable solutions (workload)"
                N_O = vec(findall(x -> x == 1, Data.IN[I,:] |> collect))
                N1_O = vec(filter(x -> x in N_O, Data.N1))
                @debug "Band 1 representatives found" num_N1=length(N1_O)
                
                workload_assigned = 0
                for n in N1_O
                    if round(value.(XY[n, del])) >= 1
                        @debug "Workload assigned" rep=n workload=value.(XY[n, del]) time_slot=del
                        workload_assigned += 1
                        XY_bar[n, del] = value.(XY[n, del])
                        XY_bar2[n, del] = value.(XY[n, del])
                    end
                end
                @info "XY solutions processed" workloads_assigned=workload_assigned
                
                @info "Updating assigned cases list"
                cases_assigned_this_iteration = 0
                for case in C_New_del
                    if round(sum(value.(X[case,:,del]))) == 1
                        Assigned_cases = vcat(Assigned_cases, case)
                        N_Assigned += 1
                        cases_assigned_this_iteration += 1
                        @debug "Case added to assigned list" case=case total_assigned=N_Assigned
                    end
                end
                @info "Cases assigned in this iteration" new_assignments=cases_assigned_this_iteration total_assigned=N_Assigned
                
                if isempty(C_New_del)
                    @debug "No more eligible cases, breaking time slot loop"
                    break
                end
            end
            @info "Completed processing all time slots for branch" branch=I
        end
        @info "Completed processing branch I=1"
        
        @info "Updating remaining cases list"
        initial_remaining = length(C_total)
        C_total = setdiff(C_total, Assigned_cases)
        @info "Cases list updated" initial_remaining=initial_remaining final_remaining=length(C_total) newly_assigned=initial_remaining-length(C_total)
    end
    
    @info "Main script completed successfully" 
    @info "Final statistics" total_assigned=N_Assigned remaining_cases=length(C_total) assignment_rate=round(N_Assigned/(N_Assigned+length(C_total))*100, digits=2)
    
    println("assigned cases " ,N_Assigned)
    
    return (
        assigned_cases = Assigned_cases,
        total_assigned = N_Assigned,
        remaining_cases = C_total,
        X_solution = X_bar2,
        Y_solution = Y_bar2,
        XY_solution = XY_bar2,
        Data = Data
    )
end

function get_reorder_mapping(original_row)
    """
    Returns the index mapping from original positions to new positions
    """
    n = length(original_row)
    
    first_nonzero_idx = findfirst(x -> x != 0 && x != 0.0, original_row)
    
    if isnothing(first_nonzero_idx)
        return collect(1:n)
    end
    
    last_nonzero_idx = findlast(x -> x != 0 && x != 0.0, original_row)
    working_segment = original_row[first_nonzero_idx:last_nonzero_idx]
    
    # Track original positions of each value
    value_positions = Dict()
    
    for (idx, val) in enumerate(working_segment)
        if val != 0 && val != 0.0
            if !haskey(value_positions, val)
                value_positions[val] = []
            end
            push!(value_positions[val], idx)
        end
    end
    
    # Get unique values in order
    unique_values = []
    seen = Set()
    
    for val in working_segment
        if val != 0 && val != 0.0 && !(val in seen)
            push!(unique_values, val)
            push!(seen, val)
        end
    end
    
    # Create mapping for the working segment
    segment_mapping = zeros(Int, length(working_segment))
    new_pos = 1
    
    for val in unique_values
        positions = value_positions[val]
        for old_pos in positions
            segment_mapping[old_pos] = new_pos
            new_pos += 1
        end
    end
    
    # Handle zeros in the working segment
    for (idx, val) in enumerate(working_segment)
        if val == 0 || val == 0.0
            segment_mapping[idx] = new_pos
            new_pos += 1
        end
    end
    
    # Create full mapping
    full_mapping = collect(1:n)
    for i in 1:length(working_segment)
        full_mapping[first_nonzero_idx + i - 1] = first_nonzero_idx + segment_mapping[i] - 1
    end
    
    return full_mapping
end

function excel_output(X_bar2, Y_bar2, XY_bar2, Data, config)
    
    working_days = config.working_days
    working_hour_per_Day = config.working_hour_per_Day
    branch_run = config.branch_run
    path = config.output_path
    @info "Starting excel_output function" timestamp=now() branch_run=branch_run working_days=working_days working_hour_per_Day=working_hour_per_Day
    
    @info "Loading rooms data from CSV"
    local data_2
    try
        data_2 = CSV.File(config.rooms_data_path, header=true,types=[Int64, Int64, String, String, String]) |> DataFrame
        @info "Rooms data loaded successfully" total_rooms=nrow(data_2)
    catch e
        @error "Failed to load rooms data" error=e file="Rooms.csv"
        rethrow(e)
	error("Failed to load rooms data (Excel output)")
    end    
    data_2 = data_2[(data_2[:,2] .== branch_run), :]
                 

    @info "Determining room range for branch"
    first_rooms = findfirst(x -> x==branch_run, data_2[:,2]) 
    Range_Rooms = first_rooms : first_rooms + size(data_2[(data_2[:,2] .== branch_run),:],1) - 1
    @info "Room range calculated" first_room=first_rooms range_size=length(Range_Rooms) Range_Rooms=Range_Rooms
    
    @info "Initializing output matrices"
    out_put_1 = Matrix{Union{Float64, String}}(undef, length(Range_Rooms), working_days*working_hour_per_Day)
    fill!(out_put_1, 0.0) 
    
    out_put_2 = Matrix{Union{Float64, String}}(undef, length(Range_Rooms), working_days*working_hour_per_Day)
    fill!(out_put_2, 0.0)
    
    out_put_3 = Matrix{Union{Float64, String}}(undef, length(Range_Rooms), working_days*working_hour_per_Day)
    fill!(out_put_3, 0.0)
    
    out_put_4 = Matrix{Union{Float64, String}}(undef, length(Range_Rooms), working_days*working_hour_per_Day)
    fill!(out_put_4, 0.0)
    
    out_put_5 = Matrix{Union{Float64, String}}(undef, length(Range_Rooms), working_days*working_hour_per_Day)
    fill!(out_put_5, 0.0)
    
    out_put_6 = Matrix{Union{Float64, String}}(undef, length(Range_Rooms), working_days*working_hour_per_Day)
    fill!(out_put_6, 0.0)
    @info "Output matrices initialized" matrix_dimensions=(length(Range_Rooms), working_days*working_hour_per_Day)
    
    @info "Processing case assignments (X_bar2) for output matrix 1 and 5"
    case_assignments = 0
    for i in Range_Rooms
        @debug "Processing room for case assignments" room=i
        seq = findfirst(x -> x==1, Data.IR[:,i])
        if isnothing(seq)
            @warn "No sequence found for room" room=i
            continue
        end
        
        for t in 1:working_days*working_hour_per_Day
            a = findfirst(x -> x >= 0.1, X_bar2[:,i,t])
            if !isnothing(a)
                @debug "Case assignment found" room=i time_slot=t case=a duration=Data.Betta[seq]
                case_assignments += 1
                out_put_1[i-first_rooms+1,t:t+Data.Betta[seq]-1] .= Float64(Data.Index_Cases[a,1])
                out_put_5[i-first_rooms+1,t:t+Data.Betta[seq]-1] .= Float64(Data.NLAF[a])

            end
        end
    end
    @info "Case assignments processing completed" total_assignments=case_assignments
    
    @info "Processing representative assignments (Y_bar2) for output matrices 2, 3, 4, and 6"
    rep_assignments = 0
    band1_assignments = 0
    band2_assignments = 0
    band3_assignments = 0
    
    for i in Range_Rooms
        @debug "Processing room for representative assignments" room=i
        seq = findfirst(x -> x==1, Data.IR[:,i])
        if isnothing(seq)
            @warn "No sequence found for room in representative processing" room=i
            continue
        end
        
        for t in 1:working_days*working_hour_per_Day
            a = findall(x -> x >= 0.1, Y_bar2[:,i,t])
            if length(a) >= 3
                @debug "Multiple representatives found" room=i time_slot=t num_reps=length(a)
                rep_assignments += 1
                
                for l in a
                    if !isnothing(findfirst(x -> x== l,Data.N1))
                        @debug "Band 1 representative assigned" room=i time_slot=t rep=l
                        out_put_2[i-first_rooms+1,t:t+Data.Betta[seq]-1] .= lpad(string(Data.data_15[l,2]), 10, '0')                            
                        out_put_6[i-first_rooms+1,t:t+Data.Betta[seq]-1] .= XY_bar2[l,t]
                        band1_assignments += 1
                    elseif !isnothing(findfirst(x -> x== l,Data.N2))
                        @debug "Band 2 representative assigned" room=i time_slot=t rep=l
                        out_put_3[i-first_rooms+1,t:t+Data.Betta[seq]-1] .= lpad(string(Data.data_15[l,2]), 10, '0')
                        band2_assignments += 1
                    elseif !isnothing(findfirst(x -> x== l,Data.N3))
                        @debug "Band 3 representative assigned" room=i time_slot=t rep=l
                        out_put_4[i-first_rooms+1,t:t+Data.Betta[seq]-1] .= lpad(string(Data.data_15[l,2]), 10, '0')
                        band3_assignments += 1
                    else
                        @warn "agent not found in any band" rep=l room=i time_slot=t
                    end
                end
            end
        end
    end
    @info "Representative assignments processing completed" total_rep_assignments=rep_assignments band1=band1_assignments band2=band2_assignments band3=band3_assignments
    
    # NEW: Reorder rows based on out_put_4
    @info "Reordering matrices based on out_put_4 pattern"
    for row_idx in 1:size(out_put_4, 1)
        @debug "Reordering row" row=row_idx
        
        # Get the reordering mapping from out_put_4
        original_row_4 = out_put_4[row_idx, :]
        mapping = get_reorder_mapping(original_row_4)
        
        # Create inverse mapping (new_pos -> old_pos)
        inverse_mapping = zeros(Int, length(mapping))
        for old_pos in 1:length(mapping)
            new_pos = mapping[old_pos]
            inverse_mapping[new_pos] = old_pos
        end
        
        # Reorder all rows using inverse mapping
        new_row_1 = similar(out_put_1[row_idx, :])
        new_row_2 = similar(out_put_2[row_idx, :])
        new_row_3 = similar(out_put_3[row_idx, :])
        new_row_4 = similar(out_put_4[row_idx, :])
        new_row_5 = similar(out_put_5[row_idx, :])
        
        for new_pos in 1:length(inverse_mapping)
            old_pos = inverse_mapping[new_pos]
            new_row_1[new_pos] = out_put_1[row_idx, old_pos]
            new_row_2[new_pos] = out_put_2[row_idx, old_pos]
            new_row_3[new_pos] = out_put_3[row_idx, old_pos]
            new_row_4[new_pos] = out_put_4[row_idx, old_pos]
            new_row_5[new_pos] = out_put_5[row_idx, old_pos]
        end
        
        # Update the matrices
        out_put_1[row_idx, :] = new_row_1
        out_put_2[row_idx, :] = new_row_2
        out_put_3[row_idx, :] = new_row_3
        out_put_4[row_idx, :] = new_row_4
        out_put_5[row_idx, :] = new_row_5
    end
    @info "Matrix reordering completed"
    
    @info "Creating DataFrames from output matrices"
    df1 = DataFrame(out_put_1, :auto)
    rename!(df1, ["$i" for i in 1:size(out_put_1, 2)])
    @debug "DataFrame 1 created" size=size(df1)
    
    df2 = DataFrame(out_put_2, :auto)
    rename!(df2, ["$i" for i in 1:size(out_put_2, 2)])
    @debug "DataFrame 2 created" size=size(df2)
    
    df3 = DataFrame(out_put_3, :auto)
    rename!(df3, ["$i" for i in 1:size(out_put_3, 2)])
    @debug "DataFrame 3 created" size=size(df3)
    
    df4 = DataFrame(out_put_4, :auto)
    rename!(df4, ["$i" for i in 1:size(out_put_4, 2)])
    @debug "DataFrame 4 created" size=size(df4)
    
    df5 = DataFrame(out_put_5, :auto)
    rename!(df5, ["$i" for i in 1:size(out_put_5, 2)])
    @debug "DataFrame 5 created" size=size(df5)
    
    df6 = DataFrame(out_put_6, :auto)
    rename!(df6, ["$i" for i in 1:size(out_put_6, 2)])
    @debug "DataFrame 6 created" size=size(df6)
    
    @info "Creating time slots for Excel headers"
    time_slots = Time(8,0,0): Minute(5) : Time(18,0,0)
    @info "Time slots created" num_slots=length(time_slots) start_time=first(time_slots) end_time=last(time_slots)
    
    @info "Creating Excel file" filename="TAX_Temp3.xlsx"
    try
        XLSX.openxlsx(path, mode="w") do xf
            @info "Excel file opened successfully"
            
            @info "Creating first sheet: Cases"
            XLSX.rename!(xf[1], "‍‍‍پرونده‌ها")
            @debug "First sheet renamed"
            
            @info "Adding time slot headers to first sheet"
            for j in 1:size(df1, 2)
                day_num = div(j-1, working_hour_per_Day) + 1
                time_slot_in_day = ((j-1) % working_hour_per_Day) + 1
                base_time_slots = Time(8,0,0) : Minute(5) : Time(18,0,0)
                
                if time_slot_in_day <= length(base_time_slots)
                    time = base_time_slots[time_slot_in_day]
                    header = "Day $day_num - $(Dates.format(time, "HH:MM"))"
                else
                    header = "Day $day_num - Slot $time_slot_in_day"
                end
                
                xf[1][XLSX.CellRef(1, j+1)] = header
            end
            @debug "Time headers added to first sheet"
            
            @info "Adding room labels to first sheet"
            for j in 1:size(df1,1)
                xf[1][XLSX.CellRef(j+1, 1)] = data_2[first_rooms + j - 1,4]
            end
            @debug "Room labels added to first sheet"
            
            @info "Populating case data in first sheet"
            for i in 1:size(df1, 1), j in 1:size(df1, 2)
                xf[1][XLSX.CellRef(i+1, j+1)] = df1[i, j]
            end
            @info "First sheet populated" rows=size(df1, 1) columns=size(df1, 2)
            
            @info "Creating second sheet: Band 1 Representatives"
            sheet2 = XLSX.addsheet!(xf, "نمایندگان بند یک")

            for j in 1:size(df2, 2)
                day_num = div(j-1, working_hour_per_Day) + 1
                time_slot_in_day = ((j-1) % working_hour_per_Day) + 1
                base_time_slots = Time(8,0,0) : Minute(5) : Time(18,0,0)
                
                if time_slot_in_day <= length(base_time_slots)
                    time = base_time_slots[time_slot_in_day]
                    header = "Day $day_num - $(Dates.format(time, "HH:MM"))"
                else
                    header = "Day $day_num - Slot $time_slot_in_day"
                end
                
                sheet2[XLSX.CellRef(1, j+1)] = header
            end
            
            for j in 1:size(df2,1)
                sheet2[XLSX.CellRef(j+1, 1)] = data_2[first_rooms + j - 1,4]
            end
            for i in 1:size(df2, 1), j in 1:size(df2, 2)
                sheet2[XLSX.CellRef(i+1, j+1)] = df2[i, j]
            end
            @info "Second sheet created and populated" rows=size(df2, 1) columns=size(df2, 2)
            
            @info "Creating third sheet: Band 2 Representatives"
            sheet3 = XLSX.addsheet!(xf, "نمایندگان بند دو")

            for j in 1:size(df3, 2)
                day_num = div(j-1, working_hour_per_Day) + 1
                time_slot_in_day = ((j-1) % working_hour_per_Day) + 1
                base_time_slots = Time(8,0,0) : Minute(5) : Time(18,0,0)
                
                if time_slot_in_day <= length(base_time_slots)
                    time = base_time_slots[time_slot_in_day]
                    header = "Day $day_num - $(Dates.format(time, "HH:MM"))"
                else
                    header = "Day $day_num - Slot $time_slot_in_day"
                end
                
                sheet3[XLSX.CellRef(1, j+1)] = header
            end
            
            for j in 1:size(df3,1)
                sheet3[XLSX.CellRef(j+1, 1)] = data_2[first_rooms + j - 1,4]
            end
            for i in 1:size(df3, 1), j in 1:size(df3, 2)
                sheet3[XLSX.CellRef(i+1, j+1)] = df3[i, j]
            end
            @info "Third sheet created and populated" rows=size(df3, 1) columns=size(df3, 2)
            
            @info "Creating fourth sheet: Band 3 Representatives"
            sheet4 = XLSX.addsheet!(xf, "نمایندگان بند سه")

            for j in 1:size(df4, 2)
                day_num = div(j-1, working_hour_per_Day) + 1
                time_slot_in_day = ((j-1) % working_hour_per_Day) + 1
                base_time_slots = Time(8,0,0) : Minute(5) : Time(18,0,0)
                
                if time_slot_in_day <= length(base_time_slots)
                    time = base_time_slots[time_slot_in_day]
                    header = "Day $day_num - $(Dates.format(time, "HH:MM"))"
                else
                    header = "Day $day_num - Slot $time_slot_in_day"
                end
                
                sheet4[XLSX.CellRef(1, j+1)] = header
            end
            
            for j in 1:size(df4,1)
                sheet4[XLSX.CellRef(j+1, 1)] = data_2[first_rooms + j - 1,4]
            end
            for i in 1:size(df4, 1), j in 1:size(df4, 2)
                sheet4[XLSX.CellRef(i+1, j+1)] = df4[i, j]
            end
            @info "Fourth sheet created and populated" rows=size(df4, 1) columns=size(df4, 2)
            
            @info "Creating fifth sheet: Case Count per LAF"
            sheet5 = XLSX.addsheet!(xf, "تعداد پرونده در هر LAF")

            for j in 1:size(df5, 2)
                day_num = div(j-1, working_hour_per_Day) + 1
                time_slot_in_day = ((j-1) % working_hour_per_Day) + 1
                base_time_slots = Time(8,0,0) : Minute(5) : Time(18,0,0)
                
                if time_slot_in_day <= length(base_time_slots)
                    time = base_time_slots[time_slot_in_day]
                    header = "Day $day_num - $(Dates.format(time, "HH:MM"))"
                else
                    header = "Day $day_num - Slot $time_slot_in_day"
                end
                
                sheet5[XLSX.CellRef(1, j+1)] = header
            end
            
            for j in 1:size(df5,1)
                sheet5[XLSX.CellRef(j+1, 1)] = data_2[first_rooms + j - 1,4]
            end
            for i in 1:size(df5, 1), j in 1:size(df5, 2)
                sheet5[XLSX.CellRef(i+1, j+1)] = df5[i, j]
            end
            @info "Fifth sheet created and populated" rows=size(df5, 1) columns=size(df5, 2)
            
            @info "Creating sixth sheet: Band 1 Representative Workload"

        end
        @info "Excel file saved successfully" filename="TAX_Temp3.xlsx"
        
    catch e
        @error "Failed to create Excel file" error=e filename="TAX_Temp3.xlsx"
        rethrow(e)
        error("Failed to create Excel file")
    end
    
    @info "excel_output function completed successfully" 
    @info "Output summary" case_assignments=case_assignments rep_assignments=rep_assignments total_sheets=6 filename="TAX_Temp3.xlsx"
end



function schedule_dadrasi(; 
    Working_Days::Int,
    #Working_Hour_Per_Day::Int,
    Branch_Run::Int,
    LowerBound::Union{String,Time},
    UpperBound::Union{String,Time},
    Objections_Data_Path::String,
    Reps_Data_Path::String,
    Rooms_Data_Path::String,
    Schedule_Data_Path::String,
    #Betta::Int,
    Juror_Time::Int,
    Output_Path::String)


    # Step 1: Create configuration
    config = SchedulingConfig(
    working_days = Working_Days,
    working_hour_per_Day = 120,
    branch_run = Branch_Run,
    LowerBound = LowerBound,
    UpperBound = UpperBound,
    main_data_path = Objections_Data_Path,
    reps_data_path = Reps_Data_Path,
    rooms_data_path = Rooms_Data_Path,
    schedule_data_path = Schedule_Data_Path,
    Betta = Int(Juror_Time),
    output_path = Output_Path
    )
    @info "Starting DadrasiSchedule workflow..."
    @info "Configuration: $(config.working_days) days, $(config.working_hour_per_Day) hours/day, branch $(config.branch_run)"

    # Step 2: Load and process all data with configuration
    @info "\n=== Step 1: Loading Data ==="
    Data = Read_data(config)
        
    @info "Data loaded successfully!"
    @info "- Total cases: $(length(Data.C_toal))"
    @info "- Total representatives: $(length(Data.N))"
    @info "- Total rooms: $(length(Data.R))"
    @info "- Time slots: $(length(Data.Del))"


    # Step 3: Run main optimization script
    @info "\n=== Step 2: Running Main Optimization ==="
    results = main_script(Data)

    if !isnothing(results)
        @info "Optimization completed successfully!"
        @info "- Cases assigned: $(results.total_assigned)"
        @info "- Cases remaining: $(length(results.remaining_cases))"
        @info "- Assignment rate: $(round(results.total_assigned / length(Data.C_toal) * 100, digits=2))%"
    else
        @info "Optimization failed or returned no results"
    end

    # Step 4: Generate Excel output (if results exist)
    if !isnothing(results)
        @info "\n=== Step 3: Generating Excel Output ==="
        excel_output(
            results.X_solution, 
            results.Y_solution, 
            results.XY_solution, 
            Data, 
            config
        )
        @info "Excel file 'results.xlsx' created successfully!"
    else
        @info "Skipping Excel output - no optimization results available"
    end

    @info "\n=== Workflow Complete ==="

end

end # module DadrasiSchedule



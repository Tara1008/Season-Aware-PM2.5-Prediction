// =====================================================
// HYDERABAD DAILY AOD + ERA5-LAND + ERA5 BLH + NDVI
// FOR CPCB STATIONS
// =====================================================


// ----------------------
// 1. SETTINGS
// ----------------------

var start = '2023-01-01';
var end   = '2023-12-31';

var stations = ee.FeatureCollection(
  'projects/urban-heat-islands-491512/assets/stations'
);

Map.centerObject(stations, 10);
Map.addLayer(stations, {}, 'Stations');

var nDays = ee.Date(end).difference(ee.Date(start), 'day');
var days  = ee.List.sequence(0, nDays.subtract(1));


// ----------------------
// 2. MODIS MAIAC AOD
//    STEP 1: Daily composite FIRST (Terra + Aqua → one image per day)
//    STEP 2: Then gap-fill across days
// ----------------------

var modisGranules = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
  .select('Optical_Depth_055')
  .filterDate(start, ee.Date(end).advance(3, 'day')) // extend slightly for edge gap-fill
  .map(function(img) {
    return img.toFloat()
      .multiply(0.001)
      .rename('AOD')
      .copyProperties(img, ['system:time_start']);
  });

// STEP 1: Collapse all granules within each day into one daily composite
var dailyAOD_raw = ee.ImageCollection(
  days.map(function(d) {
    var dayStart = ee.Date(start).advance(d, 'day');
    var dayEnd   = dayStart.advance(1, 'day');

    var daily = modisGranules
      .filterDate(dayStart, dayEnd)
      .mean(); // averages Terra + Aqua where both exist, uses whichever is available

    return daily
      .rename('AOD')
      .set('system:time_start', dayStart.millis())
      .set('date', dayStart.format('YYYY-MM-dd'));
  })
);

// STEP 2: Gap-fill across days (±3 days), now operating on clean daily images
var filledAOD = ee.ImageCollection(
  days.map(function(d) {
    var dayStart = ee.Date(start).advance(d, 'day');

    var target = ee.Image(dailyAOD_raw.toList(nDays).get(d));

    // Collect ±3 neighbouring daily images (excluding the target day itself)
    var neighbours = dailyAOD_raw
      .filterDate(
        dayStart.advance(-3, 'day'),
        dayStart.advance(4, 'day')  // 4 so the window is truly ±3
      )
      .filter(ee.Filter.neq('date', dayStart.format('YYYY-MM-dd')));

    var fill = ee.Image(
      ee.Algorithms.If(
        neighbours.size().gt(0),
        neighbours.mean(),
        target  // if no neighbours exist at all, keep original (will still be masked)
      )
    );

    return target
      .unmask(fill)   // only fills pixels that are actually masked/missing
      .rename('AOD')
      .copyProperties(target, ['system:time_start', 'date']);
  })
);


// ----------------------
// 3. ERA5-LAND: Temp, RH, Wind (surface variables, finer 0.1° resolution)
// ----------------------

var era5Land = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
  .select([
    'temperature_2m',
    'dewpoint_temperature_2m',
    'u_component_of_wind_10m',
    'v_component_of_wind_10m'
  ])
  .filterDate(start, end);

var era5LandDaily = ee.ImageCollection(
  days.map(function(d) {
    var dayStart = ee.Date(start).advance(d, 'day');
    var dayEnd   = dayStart.advance(1, 'day');

    var daily = era5Land.filterDate(dayStart, dayEnd).mean();

    // Kelvin → Celsius
    var tempC = daily.select('temperature_2m')
      .subtract(273.15)
      .rename('temp');

    var dewC = daily.select('dewpoint_temperature_2m')
      .subtract(273.15);

    // Relative Humidity via Magnus formula
    var es = tempC.expression('exp((17.625*T)/(243.04+T))', { T: tempC });
    var ed = dewC.expression('exp((17.625*Td)/(243.04+Td))', { Td: dewC });
    var rh = ed.divide(es).multiply(100).rename('humidity');

    // Wind components + derived speed
    var windU = daily.select('u_component_of_wind_10m').rename('wind_u');
    var windV = daily.select('v_component_of_wind_10m').rename('wind_v');
    var windSpeed = windU.pow(2).add(windV.pow(2)).sqrt().rename('wind_speed');

    return ee.Image.cat([tempC, rh, windU, windV, windSpeed])
      .set('system:time_start', dayStart.millis());
  })
);


// ----------------------
// 4. ERA5 STANDARD: Boundary Layer Height only (not available in ERA5-Land)
//    Using daily mean AND daily max BLH (max is often more informative for PM2.5)
// ----------------------

var era5Std = ee.ImageCollection('ECMWF/ERA5/HOURLY')
  .select(['boundary_layer_height'])
  .filterDate(start, end);

var blhDaily = ee.ImageCollection(
  days.map(function(d) {
    var dayStart = ee.Date(start).advance(d, 'day');
    var dayEnd   = dayStart.advance(1, 'day');

    var hourly = era5Std.filterDate(dayStart, dayEnd);

    var blhMean = hourly.mean().rename('blh_mean');
    var blhMax  = hourly.max().rename('blh_max');

    return ee.Image.cat([blhMean, blhMax])
      .set('system:time_start', dayStart.millis());
  })
);


// ----------------------
// 5. NDVI — MODIS MOD13A1 (500m, 16-day composite)
//    Each day is assigned the NDVI from the most recent 16-day composite
//    that started ON or BEFORE that day (forward-fill in time)
// ----------------------

var modisNDVI = ee.ImageCollection('MODIS/061/MOD13A1')
  .select('NDVI')
  .filterDate(
    ee.Date(start).advance(-16, 'day'), // go back one composite period so day 1 is covered
    end
  )
  .map(function(img) {
    return img.toFloat()
      .multiply(0.0001)   // scale factor: raw values are x10000
      .rename('NDVI')
      .copyProperties(img, ['system:time_start']);
  });

var ndviDaily = ee.ImageCollection(
  days.map(function(d) {
    var dayStart = ee.Date(start).advance(d, 'day');

    // Get the most recent composite that started on or before this day
    var mostRecent = modisNDVI
      .filterDate(
        ee.Date(start).advance(-16, 'day'),
        dayStart.advance(1, 'day')  // +1 so filterDate includes dayStart itself
      )
      .sort('system:time_start', false)  // descending → most recent first
      .first();

    return ee.Image(mostRecent)
      .rename('NDVI')
      .set('system:time_start', dayStart.millis());
  })
);


// ----------------------
// 7. JOIN ALL FOUR: AOD + ERA5-Land + BLH + NDVI
// ----------------------

var joinFilter = ee.Filter.equals({
  leftField:  'system:time_start',
  rightField: 'system:time_start'
});

// Helper to merge two ImageCollections by time
function joinCollections(primary, secondary) {
  var joined = ee.Join.inner().apply(primary, secondary, joinFilter);
  return ee.ImageCollection(
    joined.map(function(f) {
      return ee.Image(f.get('primary'))
        .addBands(ee.Image(f.get('secondary')))
        .copyProperties(ee.Image(f.get('primary')), ['system:time_start']);
    })
  );
}

var combined = joinCollections(
  joinCollections(
    joinCollections(filledAOD, era5LandDaily),  // AOD + met
    blhDaily                                     // + BLH
  ),
  ndviDaily                                      // + NDVI
);


// ----------------------
// 8. SAMPLE AT STATIONS (1.5 km buffer)
// ----------------------

var stationSamples = combined.map(function(img) {

  var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');

  var sampled = stations.map(function(station) {

    var buffer = station.geometry().buffer(1500);

    var values = img.reduceRegion({
      reducer:   ee.Reducer.mean(),
      geometry:  buffer,
      scale:     1000,
      maxPixels: 1e9
    });

    return ee.Feature(station.geometry(), values)
      .set({
        date:      date,
        station:   station.get('station'),
        latitude:  station.get('latitude'),
        longitude: station.get('longitude')
      });
  });

  return sampled;

}).flatten();


// ----------------------
// 9. REMOVE NULLS
//    Print sizes before/after so you can see how many rows are lost
// ----------------------

print('Total rows before null filter:', stationSamples.size());

var finalTable = stationSamples.filter(
  ee.Filter.notNull([
    'AOD', 'temp', 'humidity',
    'wind_u', 'wind_v', 'wind_speed',
    'blh_mean', 'blh_max',
    'NDVI'
  ])
);

print('Total rows after null filter:', finalTable.size());
print('Rows lost to nulls:', stationSamples.size().subtract(finalTable.size()));


// ----------------------
// 10. EXPORT
// ----------------------

Export.table.toDrive({
  collection:  finalTable,
  description: 'Hyderabad_Daily_AOD_ERA5_BLH_NDVI_Stations_2024',
  fileFormat:  'CSV'
});


// ----------------------
// 11. DIAGNOSTICS
// ----------------------

print('Number of stations:', stations.size());
print('Sample of output:', finalTable.limit(10));
print('Bands in combined image:', combined.first().bandNames());

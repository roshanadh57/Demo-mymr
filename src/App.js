// App.js

import React, { useState, useEffect, Suspense, lazy } from 'react';
import PatientList from './components/PatientList';
import './App.css';

const PatientChat = lazy(() => import('./components/PatientChat'));

// Use a single cache for the combined profile data
const PROFILE_CACHE_KEY = 'patientProfileCache';

function App() {
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [profileCache, setProfileCache] = useState({});

  useEffect(() => {
    // Load cache from localStorage on initial load
    try {
      const savedCache = localStorage.getItem(PROFILE_CACHE_KEY);
      if (savedCache) {
        setProfileCache(JSON.parse(savedCache));
      }
    } catch (error) {
      console.error("Failed to parse profile cache:", error);
    }
  }, []);

  const fetchAndCacheProfile = async (patientName) => {
    try {
      const response = await fetch(`http://localhost:8000/profile/${patientName}`);
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      
      const updatedCache = { ...profileCache, [patientName]: data };
      setProfileCache(updatedCache);
      localStorage.setItem(PROFILE_CACHE_KEY, JSON.stringify(updatedCache));
    } catch (error) {
      console.error("Error fetching profile:", error);
      // Cache null on error to prevent refetching
      const errorCache = { ...profileCache, [patientName]: null };
      setProfileCache(errorCache);
      localStorage.setItem(PROFILE_CACHE_KEY, JSON.stringify(errorCache));
    }
  };

  const handlePatientSelect = (patientName) => {
    setSelectedPatient(patientName);
    // Fetch profile if not already in cache
    if (!profileCache[patientName]) {
      fetchAndCacheProfile(patientName);
    }
  };

  return (
    <div className="App">
      {selectedPatient ? (
        <Suspense fallback={<div className="loading-page">Loading Patient View...</div>}>
          <PatientChat 
            patientName={selectedPatient} 
            profileData={profileCache[selectedPatient]}
            isLoading={profileCache[selectedPatient] === undefined} // Show loading if data is being fetched
            onBack={() => setSelectedPatient(null)}
          />
        </Suspense>
      ) : (
        <PatientList onPatientSelect={handlePatientSelect} />
      )}
    </div>
  );
}

export default App;
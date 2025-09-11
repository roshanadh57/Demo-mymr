import React, { useState, useEffect } from 'react';
import '../styles/PatientList.css'; // Ensure this path is correct
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {  faEye, faEdit, faTrashAlt, faSquare } from '@fortawesome/free-solid-svg-icons'; // Updated icons

function PatientList({ onPatientSelect, isSyncing }) {
  const [patients, setPatients] = useState([]);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  
  useEffect(() => {
    fetch('http://localhost:8000/patients')
      .then(res => {
        if (!res.ok) {
          throw new Error('Network response was not ok');
        }
        return res.json();
      })
      .then(data => {
        const transformedPatients = data.patients.map((patientName, index) => ({
          id: `patient-${index + 1}`,
          name: patientName,
          date: new Date(2024, 0, 15 + index * 5).toISOString().split('T')[0],
        }));
        setPatients(transformedPatients);
      })
      .catch(error => {
        console.error("Error fetching patient list:", error);
        setError('Could not fetch patient list. Please ensure the backend server is running.');
      });
  }, []);
  
  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };
  
  const handleActionClick = (actionType, patient, event) => {
    event.stopPropagation();
    if (actionType === 'view') {
      onPatientSelect(patient.name);
      return;
    }
    alert(`${actionType} for ${patient.name} (ID: ${patient.id}) - Not fully implemented yet.`);
  };
  
  const filteredPatients = patients.filter(patient =>
    patient.name.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  if (error) {
    return (
      <div className="user-management-container">
        <p className="error-message">{error}</p>
      </div>
    );
  }
  
  return (
    <div className="user-management-container">
      <h2 className="table-heading">Patient List</h2>
      <div className="search-bar-container">
        <FontAwesomeIcon icon={faSquare} className="checkbox-icon" />
        <div className="search-input-wrapper">
          <input
            type="text"
            placeholder="Search by name..."
            className="search-input"
            value={searchTerm}
            onChange={handleSearchChange}
          />
        </div>
      </div>
      <div className="table-wrapper">
        <table className="patient-table">
          <thead>
            <tr>
              <th className="th-checkbox">
                <FontAwesomeIcon icon={faSquare} className="header-checkbox-icon" />
              </th>
              <th>Name</th>
              <th>Date</th>
              <th className="th-actions">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredPatients.length > 0 ? (
              filteredPatients.map(patient => (
                <tr key={patient.id}>
                  <td className="td-checkbox">
                    <FontAwesomeIcon icon={faSquare} className="row-checkbox-icon" />
                  </td>
                  <td>{patient.name}</td>
                  <td>{patient.date}</td>
                  <td className="table-actions">
                    <button
                      className="action-button blue-button"
                      onClick={(event) => handleActionClick('view', patient, event)}
                      title="View Details"
                    >
                      <FontAwesomeIcon icon={faEye} />
                    </button>
                    <button
                      className="action-button green-button"
                      onClick={(event) => handleActionClick('edit', patient, event)}
                      title="Edit Patient"
                    >
                      <FontAwesomeIcon icon={faEdit} />
                    </button>
                    <button
                      className="action-button red-button"
                      onClick={(event) => handleActionClick('delete', patient, event)}
                      title="Delete Patient"
                    >
                      <FontAwesomeIcon icon={faTrashAlt} />
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="4" className="no-patients-found">No patients found.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {isSyncing && <p className="syncing-status">Syncing patient records...</p>}
    </div>
  );
}

export default PatientList;

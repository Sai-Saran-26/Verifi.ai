import React from 'react'
import './Contactus.css'
import leftcontimg from '../public/leftcont.png'

export const Contactus = () => {
  return (
    <>
    <div className='container'>
        <div className='cont-left'>
            {/* <img src={leftcontimg} width="330px" height="330px"/> */}
            <h1>Verifi.ai</h1>
        </div>
        <div className='cont-right'>
            <h2>Got questions or suspect a deepfake? Let’s uncover the truth together—reach out to us</h2>
            <div className="input-container">
          <input type="text" placeholder="drop ur text" className="text-input" />
          <button className="send-btn">→</button>
        </div>
        </div>
    </div>

    </>
    
  )
}
